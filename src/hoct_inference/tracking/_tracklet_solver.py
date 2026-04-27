import logging

import polars as pl
import rustworkx as rx
import tracksdata as td
from tracksdata.solvers import ILPSolver

from hoct_inference._logging import LOG


class TrackletSolver(ILPSolver):
    def _tracklet_graph(
        self,
        graph: td.graph.BaseGraph,
        tracklet_id_key: str = td.DEFAULT_ATTR_KEYS.TRACKLET_ID,
    ) -> td.graph.BaseGraph | None:
        if tracklet_id_key not in graph.node_attr_keys():
            if td.DEFAULT_ATTR_KEYS.SOLUTION not in graph.node_attr_keys():
                raise ValueError(f"`{td.DEFAULT_ATTR_KEYS.SOLUTION}` must be present in the graph")
            graph.filter(
                td.NodeAttr(td.DEFAULT_ATTR_KEYS.SOLUTION) == True,
                td.EdgeAttr(td.DEFAULT_ATTR_KEYS.SOLUTION) == True,
            ).subgraph().assign_tracklet_ids(output_key=tracklet_id_key)

        edges_df = graph.edge_attrs(
            attr_keys=[*self.edge_weight_expr.columns, td.DEFAULT_ATTR_KEYS.SOLUTION, "delta_t"]
        )
        nodes_df = graph.node_attrs(
            attr_keys=[
                td.DEFAULT_ATTR_KEYS.NODE_ID,
                td.DEFAULT_ATTR_KEYS.T,
                tracklet_id_key,
                *self.node_weight_expr.columns,
                *self.appearance_weight_expr.columns,
                *self.disappearance_weight_expr.columns,
                *self.division_weight_expr.columns,
            ]
        )

        if "edge_weight" in edges_df.columns:
            raise ValueError("`edge_weight` cannot be an existing column in the graph")

        edges_df = edges_df.with_columns(
            pl.Series(
                name="edge_weight",
                values=self._evaluate_expr(self.edge_weight_expr, edges_df),
                dtype=pl.Float64,
            )
        )

        edges_df = td.functional.join_node_attrs_to_edges(nodes_df, edges_df)

        nodes_df = nodes_df.with_columns(
            *[
                pl.Series(
                    name=f"{name}_weight",
                    values=self._evaluate_expr(expr, nodes_df),
                    dtype=pl.Float64,
                )
                for name, expr in zip(
                    ["node", "appearance", "disappearance", "division"],
                    [
                        self.node_weight_expr,
                        self.appearance_weight_expr,
                        self.disappearance_weight_expr,
                        self.division_weight_expr,
                    ],
                    strict=True,
                )
            ]
        )

        tracklet_nodes_df = (
            nodes_df.filter(
                pl.col(tracklet_id_key) >= 0,
            )
            .group_by(tracklet_id_key)
            .agg(
                pl.col(td.DEFAULT_ATTR_KEYS.T).min().alias("start_t"),
                pl.col(td.DEFAULT_ATTR_KEYS.T).max().alias("end_t"),
                pl.col("appearance_weight").sort_by(td.DEFAULT_ATTR_KEYS.T).first().alias("appearance_weight"),
                pl.col("disappearance_weight").sort_by(td.DEFAULT_ATTR_KEYS.T).last().alias("disappearance_weight"),
                pl.col("division_weight").sort_by(td.DEFAULT_ATTR_KEYS.T).last().alias("division_weight"),
                pl.col("node_weight").sort_by(td.DEFAULT_ATTR_KEYS.T).mean().alias("node_weight"),
                pl.col(td.DEFAULT_ATTR_KEYS.NODE_ID).sort_by(td.DEFAULT_ATTR_KEYS.T).first().alias("start_node_id"),
                pl.col(td.DEFAULT_ATTR_KEYS.NODE_ID).sort_by(td.DEFAULT_ATTR_KEYS.T).last().alias("end_node_id"),
            )
        )

        # pl.Config.set_tbl_cols(100)
        # pl.Config.set_tbl_rows(50)
        # print("tracklet_nodes_df")
        # print(tracklet_nodes_df)

        tracklet_nodes_df = tracklet_nodes_df.with_columns(
            (0.5 * (pl.col("start_t") + pl.col("end_t"))).round().cast(pl.Int32).alias(td.DEFAULT_ATTR_KEYS.T),
        )

        tracklet_graph = td.graph.IndexedRXGraph()
        tracklet_graph.add_node_attr_key("start_t", pl.Float32, -1.0)
        tracklet_graph.add_node_attr_key("end_t", pl.Float32, -1.0)
        tracklet_graph.add_node_attr_key("appearance_weight", pl.Float32, 0.0)
        tracklet_graph.add_node_attr_key("disappearance_weight", pl.Float32, 0.0)
        tracklet_graph.add_node_attr_key("division_weight", pl.Float32, 0.0)
        tracklet_graph.add_node_attr_key("node_weight", pl.Float32, 0.0)
        tracklet_graph.add_node_attr_key("start_node_id", pl.Int64, -1)
        tracklet_graph.add_node_attr_key("end_node_id", pl.Int64, -1)

        tracklet_graph.bulk_add_nodes(
            list(tracklet_nodes_df.drop(tracklet_id_key).iter_rows(named=True)),
            indices=tracklet_nodes_df[tracklet_id_key],
        )

        tracklet_edges_df = edges_df.group_by(
            f"source_{tracklet_id_key}",
            f"target_{tracklet_id_key}",
        ).agg(
            pl.col("edge_weight").median().alias("edge_weight"),
            pl.col(td.DEFAULT_ATTR_KEYS.SOLUTION).any().alias(td.DEFAULT_ATTR_KEYS.SOLUTION),
        )

        tracklet_edges_df = (
            tracklet_edges_df.join(
                tracklet_nodes_df.select(tracklet_id_key, "end_t").rename({"end_t": "source_end_t"}),
                left_on=f"source_{tracklet_id_key}",
                right_on=tracklet_id_key,
                how="left",
            )
            .join(
                tracklet_nodes_df.select(tracklet_id_key, "start_t").rename({"start_t": "target_start_t"}),
                left_on=f"target_{tracklet_id_key}",
                right_on=tracklet_id_key,
                how="left",
            )
            .with_columns(
                (pl.col("target_start_t") - pl.col("source_end_t")).alias("delta_t"),
            )
            .filter(pl.col("delta_t") > 0)
            .rename(
                {
                    f"source_{tracklet_id_key}": td.DEFAULT_ATTR_KEYS.EDGE_SOURCE,
                    f"target_{tracklet_id_key}": td.DEFAULT_ATTR_KEYS.EDGE_TARGET,
                }
            )
        )

        # print("tracklet_edges_df")
        # print(tracklet_edges_df.sort(td.DEFAULT_ATTR_KEYS.EDGE_SOURCE, td.DEFAULT_ATTR_KEYS.EDGE_TARGET))

        if len(tracklet_edges_df) == 0:
            return None

        tracklet_graph.add_edge_attr_key("edge_weight", pl.Float32, 0.0)
        tracklet_graph.add_edge_attr_key(td.DEFAULT_ATTR_KEYS.SOLUTION, pl.Boolean, False)
        tracklet_graph.add_edge_attr_key("delta_t", pl.Int32, -1)
        tracklet_graph.bulk_add_edges(
            list(tracklet_edges_df.iter_rows(named=True)),
        )

        tracklet_graph.summary(attrs_stats=True)

        tracklet_invalid_edges = tracklet_graph.edge_attrs().filter(pl.col("delta_t") <= 0)
        if len(tracklet_invalid_edges) > 0:
            raise ValueError(
                f"Found {len(tracklet_invalid_edges)} edges with delta_t <= 0 in "
                f"the tracklet graph.\n{tracklet_invalid_edges}"
            )

        if LOG.isEnabledFor(logging.INFO):
            summary = tracklet_graph.summary(attrs_stats=True, print_summary=False)
            LOG.info(summary)

        # must reset the tracklet ids
        graph.update_node_attrs(attrs={tracklet_id_key: -1})

        return tracklet_graph

    def solve(self, graph: td.graph.BaseGraph) -> td.graph.BaseGraph:
        tracklet_graph = self._tracklet_graph(graph)
        if tracklet_graph is None:
            if self.return_solution:
                return graph.filter(
                    td.NodeAttr(td.DEFAULT_ATTR_KEYS.SOLUTION) == True,
                    td.EdgeAttr(td.DEFAULT_ATTR_KEYS.SOLUTION) == True,
                ).subgraph()
            else:
                return None

        ilp_solver = ILPSolver(
            edge_weight=td.EdgeAttr("edge_weight"),  #  - td.EdgeAttr(td.DEFAULT_ATTR_KEYS.SOLUTION) * math.inf,
            node_weight="node_weight",
            appearance_weight="appearance_weight",
            disappearance_weight="disappearance_weight",
            division_weight="division_weight",
            return_solution=False,
        )
        ilp_solver.solve(tracklet_graph)
        edges_df = tracklet_graph.edge_attrs()
        nodes_df = tracklet_graph.node_attrs(attr_keys=[td.DEFAULT_ATTR_KEYS.NODE_ID, "start_node_id", "end_node_id"])

        edges_df = td.functional.join_node_attrs_to_edges(nodes_df, edges_df)

        edge_ids = []
        values = []
        valid_keys = graph.edge_attr_keys()

        for edge_attrs in edges_df.iter_rows(named=True):
            src, tgt = edge_attrs["source_end_node_id"], edge_attrs["target_start_node_id"]
            try:
                edge_id = graph.edge_id(src, tgt)
            except (rx.NoEdgeBetweenNodes, ValueError):
                new_edge_attrs = {
                    td.DEFAULT_ATTR_KEYS.SOLUTION: edge_attrs[td.DEFAULT_ATTR_KEYS.SOLUTION],
                    td.DEFAULT_ATTR_KEYS.MATCHED_EDGE_MASK: False,
                    td.DEFAULT_ATTR_KEYS.EDGE_DIST: 0.0,
                    "delta_t": edge_attrs["delta_t"],
                    "edge_is_gt": False,
                    "is_div": False,
                    "similarity": -1.0,
                }
                new_edge_attrs = {k: v for k, v in new_edge_attrs.items() if k in valid_keys}
                edge_id = graph.add_edge(
                    src,
                    tgt,
                    attrs=new_edge_attrs,
                )
            edge_ids.append(edge_id)
            values.append(edge_attrs[td.DEFAULT_ATTR_KEYS.SOLUTION])

        graph.update_edge_attrs(attrs={td.DEFAULT_ATTR_KEYS.SOLUTION: values}, edge_ids=edge_ids)

        if self.return_solution:
            return graph.filter(
                td.NodeAttr(td.DEFAULT_ATTR_KEYS.SOLUTION) == True,
                td.EdgeAttr(td.DEFAULT_ATTR_KEYS.SOLUTION) == True,
            ).subgraph()
