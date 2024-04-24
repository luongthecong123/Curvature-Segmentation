import open3d as o3d
# import pyransac3d as pyrsc
from plyfile import PlyData
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.mixture import GaussianMixture
from lsq_ellipse_fitting import *

def read_KH_ply(path):
    '''
    arg: 
        path: the path to the .ply file extracted from MeshLab
        file must contains 'quality' as KH (descrete mean curvature) attribute
    returns:
        vertices: np array (N,3)
        faces: np array (m,3)
        KH: np array (N,): mean curvature for each vertices 
    '''
    plydata = PlyData.read(path)
    # ele = list(plydata.elements) # print .ply keys
    # for e in ele:
    #     print(e)    
    try:
        KH = np.array(plydata['vertex']['quality'])
    except:
        KH = np.array(plydata['vertex']['KH'])
    return KH

def threshold_KH(vertices, KH, low=-0.08, high=-0.06):
    '''
    Threshold the vertices using Mean curvature
    args: 
        vertices: np array shape (N,3)
        KH: np array (N,): mean curvature for each vertices 
        low, high: float, the threshold, default: low=-0.08, high=-0.06
    returns:
        selected_idx: np array (k,): selected vertices' indices
        selected_verts: np array (k, 3): the selected vertices
    '''
    selected_idx = np.where((KH >= low)&(KH<=high)) # cool
    selected_verts = vertices[selected_idx]
    return selected_idx, selected_verts

def plane_fitting(vertices, algo ,threshold=0.01):
    '''
    fit a plane Ax + By + Cz + D = 0 to a point cloud using Ransac or SVD
    - Ransac: robust to outliers, yet the fit is pretty random
    - SVD: the plane is guaranteed to go through the center of the ring
    and gives consistant result. But affected if point clouds have 
    too much outliers
    Args:
        vertices: np array (k,3): vertices to fit plane
        algo: 'ransac' or 'SVD'
        threshold: float, For ransac, Threshold distance from the plane 
        which is considered inlier.
    Return:
        plane_params: list, the plane parameter [A, B, C, D]
    '''
    if algo=='ransac':
        s_o3d = o3d.t.geometry.PointCloud(vertices) # code is a python wrapper of C++ = faster
        plane_params, inliers = s_o3d.segment_plane(distance_threshold=threshold,
                                                ransac_n=3,
                                                num_iterations=1000)
        # Alternative: using pyrsc library, code base in Python = slower
        # plane1 = pyrsc.Plane()
        # plane_params, best_inliers = plane1.fit(vertices, threshold)
    elif algo=='SVD':
        mean_vert = np.mean(vertices, axis=0)
        centered_verts = vertices - mean_vert
        cov = np.cov(centered_verts, rowvar=False)
        U, S, Vt = np.linalg.svd(cov)
        normal = Vt[-1]
        d = -np.dot(normal, mean_vert)
        plane_params = np.append(normal, d)
    else:
        raise ValueError('Error: incorrect choice of algorithm for plane fitting !')  
    return plane_params

def check_mesh_orientation(vertices):
    x = vertices[:, 0]
    y = vertices[:, 1]
    z = vertices[:, 2]

    dx = (np.max(x) - np.min(x))
    dy = (np.max(y) - np.min(y))
    dz = (np.max(z) - np.min(z))

    idx = np.argmax(np.array([dx, dy, dz]))
    names = ['x', 'y', 'z']
    print(f"ATTENTION !!! The mesh runs along the O{names[idx]} axis")

def cut_plane(vertices, plane_params, tolerance):
    '''
    Get the indices of the vertices above a plane Ax + By + Cz + D = 0, with tolerance (0-1)
    shows that we want to take from below a bit.
    Args:
        vertices: np array (N,3), all vertices of original mesh
        plane_params: list, the plane parameter [A, B, C, D]

    '''
    A, B, C, D = plane_params
    cut_x = vertices[:, 0]
    cut_y = vertices[:, 1]
    cut_z = vertices[:, 2]

    cut_xyz = A*cut_x + B*cut_y + C*cut_z + tolerance*D*np.ones(shape=(len(cut_x),))
    cut_xyz_above_idx = np.where(cut_xyz>0)[0]
    
    return cut_xyz_above_idx

def R_Rodrigues(a, b):
    '''
    Calc rotation using R = I + v_x + (v_x)**2 * 1/(1+c)
    Args:
        a: np array (3,): vector a
        b: np array (3,): vector b
    Return:
        R: np array (3,3): the rotation matrix that transform a into b
    Usage:
        b = (R@a.T).T  # if a has shape (1,3)
        b = R@a        # if a has shape (3,)        
    '''
    a = a/(np.linalg.norm(a) + 1e-08) # void division by zero
    b = b/(np.linalg.norm(a) + 1e-08)

    v = np.cross(a, b)
    c = np.dot(a, b)

    v_x = np.array([[0, -v[2], v[1]],
                    [v[2], 0, -v[0]],
                    [-v[1], v[0], 0]])
    
    R = np.identity(3) + v_x + np.dot(v_x,v_x) * 1/(1+c)
    return R

def plane_projection(selected_verts, plane_params, R, return_center=False):
    '''
    Project vertices onto a plane Ax + By + Cz + D = 0, rotates the plane parallel to Oxz, 
    then translate. The idea is to get the 2D version of the projected points on plane.

    Args:
    selected_verts: np array (k, 3): points that are thresholded
    plane_params: list: the plane parameters [A, B, C, D]
    R: np array (3,3): rotation matrix to align the plane to be parallel to Oxz.

    Returns:
    projected_2D: np array (k, 2): projected points onto Oxz (X = (x,z))
    '''
    A, B, C, D = plane_params
    distance = (A*selected_verts[:, 0] + B*selected_verts[:, 1] + C*selected_verts[:, 2] + D).reshape(-1, 1)
    projected_verts = selected_verts - distance*np.repeat(np.array([A, B, C]).reshape(1, -1), len(selected_verts), axis=0)

    # re-align so that vertices are parallel to Oxz
    projected_2D = (R.T@(projected_verts.T)).T  
    y_translate = np.mean(projected_2D[:, 1])
    projected_2D = projected_2D - np.array([0., y_translate, 0.])
    # projected_center = np.mean(projected_2D, axis=0, keepdims=True)[0]
    projected_2D[:, 1] = 0.
    projected_2D = np.delete(projected_2D, 1, 1)
    # return projected_2D, projected_center
    return projected_2D


def ellipse_threshold(vertices, ellipse_params, tolerance=1.1, return_in=False):
    '''
    Delete points that are inside ellipse or from a slightly tolerance times bigger ellipse.
    
    Args:
        vertices: np array (k,2): 2D vertices to check if it's inside ellipse
        ellipse_params: list len=4: the parameters of the ellipse: center, width, height, phi
        return_in: returns the vertices inside this ellipse. default: False.
    Returns:
        idx: np array (m,): the dices of the remaining vertices
    '''
    center, width, height, phi = ellipse_params
    width = width*tolerance
    height = height*tolerance
    cos_a = np.cos(phi)
    sin_a = np.sin(phi)

    subtract = vertices - center

    x = subtract[:,0]
    y = subtract[:,1]

    # inout < 0 -> inside ellipse
    inout = ((cos_a*x + sin_a*y)/width)**2 + ((sin_a*x - cos_a*y)/height)**2 - 1
    if return_in: 
        idx = np.where(inout < 0) # find indices of vertices inside the ellipse
    else:
        idx = np.where(inout > 0) # find indices of vertices outside the ellipse
    return idx

def ellipses_fitting(points, visualize=False, tolerance=1.1):
    '''
    points: np array, 2D points
    '''

    # Fit inner ellipse
    reg = LsqEllipse().fit(points)
    ellipse_inner= reg.as_parameters()

    # print(f'center: {ellipse_inner[0][0]:.3f}, {ellipse_inner[0][1]:.3f}')
    # print(f'width: {ellipse_inner[1]:.3f}')
    # print(f'height: {ellipse_inner[2]:.3f}')
    # print(f'phi: {ellipse_inner[3]:.3f}')

    # Remove inner ellipse vertices
    idx = ellipse_threshold(points, ellipse_inner, tolerance)
    verts_remain = points[idx]
    # fit outer ellipse:
    reg2 = LsqEllipse().fit(verts_remain)
    ellipse_outer = reg2.as_parameters()

    if visualize:
        # plot_ellipse(points, ellipse_inner, "Fit inner ellipse")
        # plot_ellipse(verts_remain, ellipse_inner, "Remove inner vertices")
        plot_2_ellipses(points, ellipse_inner, ellipse_outer, "Two ellipses fitting")

    return ellipse_inner, ellipse_outer

def ring_GMM(ring_verts_2D, all_rings_idx, nb_rings=3, cov_types = ["diag", "full"]):
    '''
    ring_verts_2D: np array (k, 2), 2D vertices
    all_rings_idx: np array (k,), the indices of ring vertices wrt full verts
    nb_rings: number rings: 1, 2 or 3

    returns:
        ring_clusters_idx: list of np array containing the idx of clusters wrt the full verts (vertices)
    '''
    colors = ["navy", "turquoise", "darkorange"]
    ring_names = ["ring 1", "ring 2", "ring 3"]
    # cov_types = ["spherical", "diag", "tied", "full"]
    cov_types = ["diag", "full"]
    # gm = GaussianMixture(n_components=3, random_state=0).fit(vertices)

    estimators = {
        cov_type: GaussianMixture(
            n_components=nb_rings, covariance_type=cov_type, max_iter=20, random_state=0
        )
        for cov_type in cov_types
    }

    n_estimators = len(estimators)

    three_rings_ellipses = [] # a list of lists of lists

    for index, (name, estimator) in enumerate(estimators.items()):

        # Train the other parameters using the EM algorithm.
        estimator.fit(ring_verts_2D)

        h = plt.subplot(2, n_estimators // 2, index + 1)
        
        ellipses_params = make_ellipses(estimator, h, colors)

        three_rings_ellipses.append(ellipses_params)

        for n, color in enumerate(colors):
            data = ring_verts_2D
            plt.scatter(
                data[:, 0], data[:, 1], s=0.8, color=color, label=ring_names[n]
            )

        # y_train_pred = estimator.predict(vertices)

        plt.xticks(())
        plt.yticks(())
        plt.title(name)

    plt.legend(scatterpoints=1, loc="lower right", prop=dict(size=12))

    plt.show()

    ring_clusters_idx = []

    for i, type_cov in enumerate(three_rings_ellipses):
        type_cov_idx_lst = []
        for j, ellipse_param in enumerate(type_cov):
            ring_idx = ellipse_threshold(ring_verts_2D, ellipse_param, tolerance=1.5, return_in=True)
            idx_wrt_origin = all_rings_idx[ring_idx[0]]
            type_cov_idx_lst.append(idx_wrt_origin)
        ring_clusters_idx.append(type_cov_idx_lst)
    #         plt.subplot(130 + j + 1)
    #         plt.scatter(ring[:,0], ring[:,1])
    #         plt.title(ring_names[j])
    #         plt.axis('equal')
    #     # plt.suptitle("Covariance type: " + cov_types[i])
    #     plt.tight_layout(pad=0, w_pad=2)
    #     plt.savefig("segmented_rings_" + cov_types[i]+".png", 
    #                 dpi=300, pad_inches=0)
    #     plt.show()
            
    return ring_clusters_idx

def zero_middle(array):
    return np.dstack((array, np.zeros_like(array))).reshape(array.shape[0], -1)[:,:3]

#################################################################################
############################## Visualization ####################################
#################################################################################

def vis_points(vertices):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)

    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd])

def get_plane(plane_params):
    '''
    create a meshgrid for the plane Ax + By + Cz + D = 0
    '''
    A, B, C, D = plane_params
    calc_z = lambda x, y: (-A*x - B*y - D)/C
    x = np.linspace(-10, 110, 200)
    y = np.linspace(-10, 110, 200)
    z = calc_z(x,y).astype(np.float16)
    xx, yy = np.meshgrid(x, y)
    zz = calc_z(xx, yy).astype(np.float16)
    plane_verts = np.stack((xx.flatten(), yy.flatten(), zz.flatten())).T

    return plane_verts

def plot_ellipse(vertices, ellipse_params, title):
    '''
    draw vertices and fitted ellipse on the same plot
    Args:
    vertices: np array (k,2): 2D projected vertices
    ellipse_params: list len=4: center, width, height, phi
    (Not that width will be multiplied by 2 for plotting with plt)
    Return: None
    '''
    center, width, height, phi = ellipse_params

    fig = plt.figure(figsize=(6, 6))
    ax = plt.subplot()
    ax.axis('equal')
    ax.plot(vertices[:, 0], vertices[:, 1], 'ro', zorder=1)
    ellipse = matplotlib.patches.Ellipse(
        xy=center, width=2*width, height=2*height, angle=np.rad2deg(phi),
        edgecolor='b', fc='None', lw=2, label='Fit', zorder=2
    )
    ax.add_patch(ellipse)

    plt.xlabel('$X_1$')
    plt.ylabel('$X_2$')
    plt.legend()
    plt.title(title)
    plt.show()    

def plot_2_ellipses(vertices, ellipse_params_1, ellipse_params_2, title):
    '''
    draw vertices and fitted ellipses on the same plot
    Args:
    vertices: np array (k,2): 2D projected vertices
    ellipse_params: list, len=4: center, width, height, phi
    (Not that width will be multiplied by 2 for plotting with plt)
    Return: None
    '''
    center, width, height, phi = ellipse_params_1

    fig = plt.figure(figsize=(6, 6))
    ax = plt.subplot()
    ax.axis('equal')
    ax.plot(vertices[:, 0], vertices[:, 1], 'ro', zorder=1)
    ellipse1 = matplotlib.patches.Ellipse(
        xy=center, width=2*width, height=2*height, angle=np.rad2deg(phi),
        edgecolor='b', fc='None', lw=2, label='Fit', zorder=2
    )
    ax.add_patch(ellipse1)

    center, width, height, phi = ellipse_params_2

    ellipse2 = matplotlib.patches.Ellipse(
        xy=center, width=2*width, height=2*height, angle=np.rad2deg(phi),
        edgecolor='b', fc='None', lw=2, label='Fit', zorder=2
    )
    
    ax.add_patch(ellipse2)
    plt.xlabel('$X_1$')
    plt.ylabel('$X_2$')
    plt.legend()
    plt.title(title)
    plt.show()  


def make_ellipses(gmm, ax, colors):
    '''
    Draw ellipses using GMM means and covariances
    return: ellipses_param: a list of lists.
    3 lists. Each list contains 4 params: center, width, height, phi
    '''
    ellipses_params =[]
    for n, color in enumerate(colors):
        if gmm.covariance_type == "full":
            covariances = gmm.covariances_[n][:2, :2]
        elif gmm.covariance_type == "tied":
            covariances = gmm.covariances_[:2, :2]
        elif gmm.covariance_type == "diag":
            covariances = np.diag(gmm.covariances_[n][:2])
        elif gmm.covariance_type == "spherical":
            covariances = np.eye(gmm.means_.shape[1]) * gmm.covariances_[n]
        v, w = np.linalg.eigh(covariances)
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
        ell = matplotlib.patches.Ellipse(
            gmm.means_[n, :2], v[0], v[1], angle=180 + angle, color=color
        )
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.5)
        ax.add_artist(ell)
        ax.set_aspect("equal", "datalim")
        params = [gmm.means_[n, :2], v[0]/2, v[1]/2, np.deg2rad(180+angle)]
        ellipses_params.append(params)
    return ellipses_params