#########################################################
#Princess Margaret Cancer Research Tower
#Schwartz Lab
#Javier Ruiz Ramirez
#November 2024
#########################################################
#Questions? Email me at: javier.ruizramirez@uhn.ca
#########################################################

import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import shapely.geometry as geo
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
from collections import defaultdict
from scipy.spatial import voronoi_plot_2d
from scipy.spatial import ConvexHull
from typing import Optional

#Matplotlib parameters.
mpl.use("agg")
mpl.rcParams["figure.dpi"]=600
# mpl.rcParams["pdf.fonttype"]=42
mpl.rc("pdf", fonttype=42)

font = {'weight' : 'normal', 'size'   : 18}
mpl.rc("font", **font)

class CellMetrik:
    """
    Geometric manipulations on a "cell" embedding.
    """

    #=================================================
    def __init__(self, 
                 input: pd.DataFrame,
                 output: str,
    ):
        """
        """
        # self.path_to_embedding = input
        df = self.read_coordinate_based_embedding(input)
        self.points = df.to_numpy()

        self.output = os.path.abspath(output)
        os.makedirs(self.output, exist_ok=True)

        self.voronoi: Voronoi = None
        self.cell_to_area = defaultdict(float)
        self.cell_to_n_edges = defaultdict(int)
        self.cells_with_infinite_area = set()
        self.allow_incremental = True
        self.convex_hull = None

    #=================================================
    def read_coordinate_based_embedding(self,
                                        path: str):
        """
        Load the embedding as a data frame, where the 
        first column has the indices.
        """
        df = pd.read_csv(path, index_col=0)

        #We do this to select only one batch and 
        #make the computations faster for testing
        #purposes.
        df_clone = df.copy()
        df_clone["batch"] = df.index.str[-9:]
        mask = df_clone["batch"] == "175999-FT"
        df = df.loc[mask].copy()

        return df
    #=====================================
    def extract_convex_hull_from_voronoi_regions(self):
        """
        """
        convex_hull = []
        self.cells_with_infinite_area = set()

        for point_index, region_index in enumerate(
            self.voronoi.point_region):
            region = self.voronoi.regions[region_index]
            if -1 in region:
                # This region has infinite area.
                point = self.voronoi.points[point_index]
                convex_hull.append(point)
                self.cells_with_infinite_area.add(point_index)
        
        self.convex_hull = ConvexHull(convex_hull)
        
    #=====================================
    def extract_convex_hull_from_voronoi_ridges(self):
        """
        """
        convex_hull = set()

        it = zip(self.voronoi.ridge_points,
                 self.voronoi.ridge_vertices)

        for ridge_point, ridge_vertex in it:

            v_0, v_1 = ridge_vertex

            if v_0 < 0 or v_1 < 0:
                # has_infinite_area = True
                for p in ridge_point:
                    convex_hull.add(p)
            else:
                continue

        hull = list(convex_hull)

        points = self.voronoi.points[hull]
        self.convex_hull = ConvexHull(
            points,
            incremental=self.allow_incremental)

            

    #=====================================
    def add_bounding_box_to_voronoi(self):
        """
        """

        # delta = np.std(self.points, axis=0) / 2
        delta = np.std(self.points, axis=0)

        x_min = self.points[:,0].min() - delta[0]
        x_max = self.points[:,0].max() + delta[0]

        y_min = self.points[:,1].min() - delta[1]
        y_max = self.points[:,1].max() + delta[1]

        down_left = np.array([x_min, y_min])
        down_right = np.array([x_max, y_min])
        up_left = np.array([x_min, y_max])
        up_right = np.array([x_max, y_max])

        down_mid = (down_left + down_right) / 2
        up_mid = (up_left + up_right) / 2
        left_mid = (up_left + down_left) / 2
        right_mid = (up_right + down_right) / 2

        corners = [down_left, down_right, up_left, up_right]
        mid_points = [down_mid, up_mid, left_mid, right_mid]

        new_points = np.array(corners + mid_points)
        new_points = new_points.reshape(-1,2)

        self.voronoi.add_points(new_points)
        # points = np.vstack((self.points, extra))
        
        # self.points = points

    #=====================================
    def create_voronoi_diagram(self):
        """
        """
        self.voronoi = Voronoi(
            self.points,
            incremental=self.allow_incremental)

    #=====================================
    def create_voronoi_from_square_grid(self):
        """
        """
        points = np.array([[0, 0], [0, 1],[0, 2],
                           [1, 0], [1, 1], [1, 2],
                           [2, 0], [2, 1], [2, 2]])

        self.voronoi = Voronoi(
            points,
            incremental=self.allow_incremental)

    #=====================================
    def plot_voronoi_diagram(self, fname: str):
        """
        """
        fig, ax = plt.subplots()
        ax.set_xlim(-30,30)
        ax.set_ylim(-30,30)
        voronoi_plot_2d(self.voronoi,
                        ax=ax,
                        show_vertices=False,
                        show_points=True,
                        line_colors="orange",
                        line_width=0.5,
                        point_size=0.5,
        )
        ax.set_xlim(-30,30)
        ax.set_ylim(-30,30)

        fname = f"voronoi_{fname}.pdf"
        fname = os.path.join(self.output, fname)
        fig.savefig(fname, bbox_inches="tight")

    #=================================================
    def compute_metadata_for_voronoi_diagram(self):
        """
        This function computes the area and the 
        number of faces of each Voronoi cell.

        We call the points of the Voronoi diagram
        vertices. The points of the original set
        are simply called points.

        The ridge points define edges in terms
        of the original points. These edges are
        perpendicular to the edges defined by
        the ridge_vertices. The ridge vertices
        are edges of the Voronoi diagram.

        The area of a Voronoi cell is computed by 
        partitioning it into triangles. Let E = (v0,v1)
        be an edge of the Voronoi diagram. We consider the 
        two cells that share that edge. Let one cell
        correspond to the point p0 and another to the
        point p1. Then, for either Voronoi cell, 
        the area due to the edge E is base * height / 2,
        where the base is equal to ||v0-v1|| and
        the height is equal to ||p0-p1|| / 2.

        -----------------v0---------------
        |              . |.              |
        |            .   |  .            |
        |          .     |    .          |
        |        .       |      .        |
        |       p0...............p1      |
        |        .       |       .       |
        |         .      |      .        |
        |          .     |     .         |
        |           .    |    .          |
        |            .   |   .           |
        |             .  |  .            |
        |              . | .             |
        |               .|.              |
        -----------------v1---------------

        For the case of an open Voronoi cell, i.e.,
        one with infinite area, we do the following.
        -----------------v0---------------
        |              . | .              |
        |            .   |   .            |
        |          .     |     .          |
        |        .       |       .        |
        |       p0...............p1       |
        |        .       |       .        |
        |          .     |     .          |
        |            .   |   .            |
        |              . | .              |
        |                |                |
        |                |                |
        |                |                |
        |                inf              |

        If one vertex is located at infinity,
        we find the distance D between the other vertex
        and the midpoint between the two points that 
        define the line segment perpendicular to the
        the edge. We use 2 times this distance as
        the base of the triangle 
        """
        it = zip(self.voronoi.ridge_points,
                 self.voronoi.ridge_vertices)

        for ridge_point, ridge_vertex in it:


            p_0, p_1 = ridge_point
            v_0, v_1 = ridge_vertex
            p_0 = self.voronoi.points[p_0]
            p_1 = self.voronoi.points[p_1]
            midpoint = (p_0 + p_1) / 2

            height = np.linalg.norm(p_0 - p_1) / 2
            # The height of the triangle, which equals
            # the distance from the edge to any point,
            # or, equivalently, half the distance between
            # the points.

            has_infinite_area = False

            if v_0 < 0:
                # v_0 = np.inf
                vertex = self.voronoi.vertices[v_1]
                d = np.linalg.norm(vertex - midpoint)
                base = 2*d
                has_infinite_area = True
            elif v_1 < 0:
                # v_1 = np.inf
                vertex = self.voronoi.vertices[v_0]
                d = np.linalg.norm(vertex - midpoint)
                base = 2*d
                has_infinite_area = True
            else:
                v_0 = self.voronoi.vertices[v_0]
                v_1 = self.voronoi.vertices[v_1]
                base = np.linalg.norm(v_0 - v_1)
            
            triangle_area = base * height / 2

            for p in ridge_point:
                self.cell_to_area[p] += triangle_area
                self.cell_to_n_edges[p] += 1
                if has_infinite_area:
                    self.cells_with_infinite_area.add(p)


    #=====================================
    def plot_voronoi_area(self, fname: str):
        """
        """
        fig, ax = plt.subplots()
        # ax.set_xlim(-30,30)
        # ax.set_ylim(-30,30)
        voronoi_plot_2d(self.voronoi,
                        ax=ax,
                        show_vertices=False,
                        show_points=True,
                        line_colors="orange",
                        line_width=0.5,
                        point_size=0.5,
        )
        ax.set_xlim(-30,30)
        ax.set_ylim(-30,30)

        fname = f"voronoi_{fname}.pdf"
        fname = os.path.join(self.output, fname)
        fig.savefig(fname, bbox_inches="tight")
    #=====================================
    def compute_convex_hull_from_points(self):
        """
        """
        self.convex_hull = ConvexHull(
            self.points,
            incremental=self.allow_incremental)

    #=====================================
    def plot_convex_hull(self, fname: str):
        """
        """
        fig, ax = plt.subplots()
        ax.set_xlim(-30,30)
        ax.set_ylim(-30,30)
        voronoi_plot_2d(self.voronoi,
                        ax=ax,
                        show_vertices=False,
                        show_points=True,
                        line_colors="orange",
                        line_width=0.5,
                        point_size=0.5,
        )
        ax.set_xlim(-30,30)
        ax.set_ylim(-30,30)

        for simplex in self.convex_hull.simplices:
            x = self.points[simplex,0]
            y = self.points[simplex,1]
            ax.plot(x,y,"k-", linewidth=1)


        fname = f"voronoi_{fname}.pdf"
        fname = os.path.join(self.output, fname)
        fig.savefig(fname, bbox_inches="tight")

    #=====================================
    def intersect_voronoi_with_convex_hull(
            self,
            use_all_cells: bool = False,
            fname: str = "",
    ):
        """
        """
        vertices = self.convex_hull.vertices
        points = self.convex_hull.points[vertices]
        g_points = geo.MultiPoint(points)
        convex_hull = g_points.convex_hull

        if 0 < len(fname):
            fig, ax = plt.subplots()

        for point_index, region_index in enumerate(
            self.voronoi.point_region):
            region = self.voronoi.regions[region_index]

            if -1 in region:
                continue

            polygon_np = self.voronoi.vertices[region]
            x = polygon_np[:,0]
            y = polygon_np[:,1]
            polygon = np.vstack((polygon_np,
                                 polygon_np[0]))
            polygon = geo.Polygon(polygon)

            cond=point_index in self.cells_with_infinite_area

            if use_all_cells or cond:

                polygon = polygon.intersection(
                    convex_hull)
                x = polygon.boundary.coords.xy[0][:-1]
                y = polygon.boundary.coords.xy[1][:-1]
                polygon_np = np.reshape((x,y),(-1,2))

            area = polygon.area

            if 0 < len(fname):
                ax.fill(x, y, alpha=0.8)

        if 0 < len(fname):
            fname = f"voronoi_{fname}.pdf"
            fname = os.path.join(self.output, fname)
            fig.savefig(fname, bbox_inches="tight")

    #=====================================

