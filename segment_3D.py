# least-squares-ellipse-fitting (Radim Halir and Jan Flusser, 1998)
from utilities import *


# plt.rcParams['text.usetex'] = True # Latex plot

mesh_path = "mesh/3_rings.ply"
# mesh_path = "mesh/1_ring.ply"

KH = read_KH_ply(mesh_path)
mesh_o3d = o3d.io.read_triangle_mesh(mesh_path)
vertices = np.asarray(mesh_o3d.vertices)
low_KH = -0.08
high_KH = -0.03
plane_thresh = 0.02

all_rings_idx, all_rings_verts = threshold_KH(vertices, KH, low=low_KH, high=high_KH)
all_rings_idx = all_rings_idx[0]
plane_params_all = plane_fitting(all_rings_verts, algo ='SVD' ,threshold=plane_thresh)

check_mesh_orientation(vertices)
# CAUTION!!! rotation matrix that transform n = (A,B,C) into Oy, only applies when 
# mesh orientation runs along Oz plane.
R = R_Rodrigues(plane_params_all[:3], np.array([0.,1.,0.]))

##### Run GMM on all the mean gaussian thresholded vertices #####

all_rings_verts_2D = plane_projection(all_rings_verts, plane_params_all, R)
ring_clusters_idx = ring_GMM(all_rings_verts_2D, all_rings_idx, nb_rings=3)
ring_clusters_idx = ring_clusters_idx[1] # select 2-nd method in ["diag", "full"]

# visualize the ring in mesh
colors = np.zeros_like(vertices)
colors.fill(0.5)

# vis_points(selected_verts)
for ring_verts_idx in ring_clusters_idx:

    ring_verts = vertices[ring_verts_idx]
    # fit a plane to one ring
    plane_params = plane_fitting(ring_verts, algo ='SVD' ,threshold=plane_thresh)

    # CAREFUL!!! rotation matrix that transform n = (A,B,C) into Oy, incase our body
    R = R_Rodrigues(plane_params[:3], np.array([0.,1.,0.]))

    # project all points onto the plane
    all_projected_2D = plane_projection(vertices, plane_params, R)
    
    # select the ring vertices
    ring_verts_2D = all_projected_2D[ring_verts_idx]

    ellipse_inner, ellipse_outer = ellipses_fitting(ring_verts_2D, visualize=True, tolerance=1.0)

    indices_1 = ellipse_threshold(all_projected_2D, ellipse_outer, tolerance=1.05, return_in=True) # inside of ellipse_outer
    indices_2 = ellipse_threshold(all_projected_2D, ellipse_inner, tolerance=0.95, return_in=False) # outside of ellipse_inner

    # NEED MORE TEST!!! the line below can (with ransac) give us the vertices below the plane in case the plane's normal is upside down
    indices_3 = cut_plane(vertices, plane_params, tolerance=0.9)
    in_between_idx = np.intersect1d(indices_1, indices_2)
    above_idx = np.intersect1d(in_between_idx, indices_3) # Final result, the indices of the segmented ring

    colors[in_between_idx] = [0., 1., 0.] # changing segmented into green
    # colors[above_idx] = [0., 1., 0.] # changing segmented into green

mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(colors)    
mesh_o3d.compute_vertex_normals()
o3d.visualization.draw_geometries([mesh_o3d])

# in_between_idx is the indices of our results (in the for loop)
# could work if you only care about the side closest to the camera.

# above_idx is the indices of our results if only keep parts 
# above the plane (in the for loop) - sometimes cut the wrong side
# (less stable)