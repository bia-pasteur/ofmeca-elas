"""Creates the gmsh mesh for finite elements computations"""


from typing import Callable
import gmsh # type: ignore
import numpy as np
import cv2
from mpi4py import MPI
from dolfinx import io
from dolfinx.io import XDMFFile
from dolfinx.io import gmsh as gmshio
import dolfinx
import pyvista

def gmsh_sphere(
    model: gmsh.model, 
    name: str, 
    physical_length: float
) -> gmsh.model:
    """Create a Gmsh model of a sphere.

    Args:
        model (gmsh.model): Gmsh model to add the mesh to.
        name (str): Name (identifier) of the mesh to add.
        L_phys (float) : Radius of the sphere

    Returns:
        gmsh.model : Gmsh model with a sphere mesh added.

    """
    model.add(name)
    model.setCurrent(name)
    
    # The radius is fixed to L_phys
    sphere = model.occ.addSphere(0, 0, 0, physical_length, tag=1)

    # Synchronize OpenCascade representation with gmsh model
    model.occ.synchronize()

    # Add physical marker for cells. It is important to call this
    # function after OpenCascade synchronization
    surfaces = model.getBoundary([(3, sphere)], oriented=False)
    surface_tags = [s[1] for s in surfaces]
    
    # Define physical group for the volume (3D entities)
    model.add_physical_group(dim=3, tags=[sphere], tag=1)
    model.set_physical_name(dim=3, tag=1, name="SphereVolume")
    
    model.add_physical_group(dim=2, tags=surface_tags, tag=2)
    model.set_physical_name(dim=2, tag=2, name="SphereSurface")

    lc = 0.1*physical_length
    model.mesh.setSize(model.getEntities(0), lc)
    model.mesh.generate(dim=3)
    return model
    
            
def gmsh_disk(
    model: gmsh.model,
    name: str,
    physical_length: float
):
    '''Create a Gmsh model of a disk.

    Args:
        model (gmsh.model): Gmsh model to add the mesh to.
        name (str): Name (identifier) of the mesh to add.
        physical_length (float) : Radius of the disk

    Returns:
        gmsh.model : Gmsh model with a sphere mesh added.

    '''
    model.add(name)
    model.setCurrent(name)

    # The radius is fixed to L_phys
    disk = gmsh.model.occ.addDisk(0, 0, 0, physical_length, 1)

    # Synchronize OpenCascade representation with gmsh model
    model.occ.synchronize()
    
    # Define physical group for the surface (2D entities)
    model.addPhysicalGroup(2, [disk], 1)
    model.setPhysicalName(2, 1, "DiskSurface")
    
    # Create smaller 
    lc = 0.1*physical_length
    model.mesh.setSize(model.getEntities(0), lc)
    
    model.mesh.generate(dim=2)
    
    return model


def create_cell_like_shape(num_points, base_radius, noise_amplitude, num_fourier_modes, rng, r_min_ratio=0.5):
    """
    Generates a 2D cell-like boundary without self-crossing.
    Thanks to Samuel Dubos for the inspiration.

    Args:
        num_points (int): Number of boundary points.
        base_radius (float): Average radius of the cell.
        noise_amplitude (float): Amplitude of Fourier noise.
        num_fourier_modes (int): Number of Fourier modes for perturbation.
        r_min_ratio (float): Minimum allowed radius as fraction of base_radius.

    Returns:
        x_coordinates, y_coordinates (np.ndarray): Ordered coordinates of the boundary.
    """
    angles = np.linspace(0, 2*np.pi, num_points, endpoint=False)
    fourier_noise = np.zeros(num_points)

    # Generate smooth Fourier noise
    for k in range(1, num_fourier_modes+1):
        phase = rng.uniform(0, 2*np.pi)
        amplitude = rng.normal(0, 1) / k
        fourier_noise += amplitude * np.sin(k*angles + phase)

    # Perturbed radius
    radius_values = base_radius + noise_amplitude * fourier_noise

    # Clamp to avoid negative or too small radii
    r_min = base_radius * r_min_ratio
    radius_values = np.maximum(radius_values, r_min)
    
    # Convert to Cartesian
    x = radius_values * np.cos(angles)
    y = radius_values * np.sin(angles)

    # Sort by polar angle to avoid crossings
    theta = np.arctan2(y, x)
    sort_idx = np.argsort(theta)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]

    return x_sorted, y_sorted


def gmsh_cell_shape(
    model: gmsh.model,
    name: str,
    physical_length: float,
    rng,
    num_points: int = 100,
    noise_amplitude: float = 0.2,
    num_fourier_modes: int = 5,
    lc: float = None
) -> gmsh.model:
    """
    Create a Gmsh model of a random cell-like shape.

    Args:
        model (gmsh.model): Gmsh model to add the mesh to.
        name (str): Name (identifier) of the mesh to add.
        physical_length (float): Base radius of the shape.
        num_points (int): Number of boundary discretization points.
        noise_amplitude (float): Amplitude of Fourier noise.
        num_fourier_modes (int): Number of Fourier modes for perturbation.
        lc (float): Characteristic mesh length (default: 0.1 * physical_length).

    Returns:
        gmsh.model: Gmsh model with a random shape mesh added.
    """
    model.add(name)
    model.setCurrent(name)

    x_coords, y_coords = create_cell_like_shape(num_points, physical_length, noise_amplitude, num_fourier_modes, rng, lc)

    if lc is None:
        lc = 0.1 * physical_length

    points = [model.geo.addPoint(x, y, 0, lc) for x, y in zip(x_coords, y_coords)]

    lines = [model.geo.addLine(points[i], points[(i+1) % len(points)]) for i in range(len(points))]

    cl = model.geo.addCurveLoop(lines)
    surface = model.geo.addPlaneSurface([cl])

    model.geo.synchronize()

    model.addPhysicalGroup(2, [surface], 1)
    model.setPhysicalName(2, 1, "CellSurface")

    model.mesh.generate(2)

    return model, x_coords, y_coords


def gmsh_cell_from_image(
    img: np.ndarray,
    model: gmsh.model,
    name: str
) -> gmsh.model:
    """
    Create a Gmsh model of a cell from an image.

    Args:
        img (np.ndarray): Image of a masked cell
        model (gmsh.model): Gmsh model to add the mesh to.
        name (str): Name (identifier) of the mesh to add.

    Returns:
        gmsh.model: Gmsh model with a mesh of a cell added.
    """
    # Settings of the model
    model.add(name)
    model.setCurrent(name)
    
    # Get the coordinated of the contours of the cell
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = np.array(contours)
    x_coords = contours[0, :, 0, 0]
    y_coords = contours[0, :, 0, 1]
    
    # Create the outline of the mesh using the coordinates
    points = [model.geo.addPoint(x, y, 0, 0) for x, y in zip(x_coords, y_coords)]
    lines = [model.geo.addLine(points[i], points[(i+1) % len(points)]) for i in range(len(points))]
    cl = model.geo.addCurveLoop(lines)
    surface = model.geo.addPlaneSurface([cl])

    # Create the model
    model.geo.synchronize()
    model.addPhysicalGroup(2, [surface], 1)
    model.setPhysicalName(2, 1, "CellSurface")

    model.mesh.generate(2)

    return model, x_coords, y_coords


def create_mesh(
    comm: MPI.Comm,
    model: gmsh.model,
    name: str,
    filename: str,
    mode: str
):
    """Create a DOLFINx from a Gmsh model and output to file.

    Args:
        comm (MPI.Comm): MPI communicator top create the mesh on.
        model (gmsh.model): Gmsh model.
        name (str): Name (identifier) of the mesh to add.
        filename (str): XDMF filename.
        mode (str): XDMF file mode. "w" (write) or "a" (append).

    """
    mesh_data = gmshio.model_to_mesh(model, comm, rank=0)
    
    mesh_data.mesh.name = name
    if mesh_data.cell_tags is not None:
        mesh_data.cell_tags.name = f"{name}_cells"
    if mesh_data.facet_tags is not None:
        mesh_data.facet_tags.name = f"{name}_facets"
    if mesh_data.ridge_tags is not None:
        mesh_data.ridge_tags.name = f"{name}_ridges"
    if mesh_data.peak_tags is not None:
        mesh_data.peak_tags.name = f"{name}_peaks"
    with XDMFFile(mesh_data.mesh.comm, filename, mode) as file:
        mesh_data.mesh.topology.create_connectivity(mesh_data.mesh.topology.dim - 1, mesh_data.mesh.topology.dim) 
        file.write_mesh(mesh_data.mesh)
        if mesh_data.cell_tags is not None:
            file.write_meshtags(
                mesh_data.cell_tags,
                mesh_data.mesh.geometry,
                geometry_xpath=f"/Xdmf/Domain/Grid[@Name='{name}']/Geometry",
            )
        if mesh_data.facet_tags is not None:
            file.write_meshtags(
                mesh_data.facet_tags,
                mesh_data.mesh.geometry,
                geometry_xpath=f"/Xdmf/Domain/Grid[@Name='{name}']/Geometry",
            )
        if mesh_data.ridge_tags is not None:
            file.write_meshtags(
                mesh_data.ridge_tags,
                mesh_data.mesh.geometry,
                geometry_xpath=f"/Xdmf/Domain/Grid[@Name='{name}']/Geometry",
            )
        if mesh_data.peak_tags is not None:
            file.write_meshtags(
                mesh_data.peak_tags,
                mesh_data.mesh.geometry,
                geometry_xpath=f"/Xdmf/Domain/Grid[@Name='{name}']/Geometry",
            )

def create_mesh_file(
    mesh_function: Callable,
    img: np.ndarray=None,
    physical_length: float=None,
    rng=None,
    num_points=None,
    noise_amplitude=None,
    num_fourier_modes=None,
    lc=None
) -> dolfinx.mesh.Mesh:
    """    
    Creates a computational mesh using a user-defined mesh function,
    writes it to XDMF, and reads it back as a DolfinX mesh.

    Args:
        mesh_function (Callable): Function that generates the Gmsh model.
        img (np.ndarray, optional): Image of a masked cell if we create the mesh from a masked cell. Defaults to None.
        physical_length (float, optional): Physical length of the cell if we create the mesh from a random shape from scratch. Defaults to None.
        rng (random.generator, optional): random generator. Defaults to None.
        num_points (int, optional): Number of points to create the mesh if we create it from a random shape from scratch. Defaults to None.
        noise_amplitude (float, optional): Amplitude of the noise to add to the original disk if we create the mesh from a random shape from scratch. Defaults to None.
        num_fourier_modes (int, optional): Number of Fourier modes on the original disk if we create the mesh from a random shape from scratch. Defaults to None.
        lc (float, optional): Minimum allowed radius as fraction of physical_length. Defaults to None.

    Returns:
        dolfinx.mesh.Mesh: The generated computational mesh ready for FEM simulation.
    """
    # Initialize gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)  # 1 to enable terminal output

    # Create model
    model_ = gmsh.model()
    
    if img is not None:
        model, x_coords, y_coords = mesh_function(img, model_, "Grid")
    
    else :  
        model, x_coords, y_coords = mesh_function(model_, "Grid", physical_length, rng, num_points, noise_amplitude, num_fourier_modes, lc)
    
    # Fix the order of the elements // à préciser
    model.mesh.set_order(1)

    # Create the file
    create_mesh(MPI.COMM_SELF, model, "Grid", "out_gmsh/mesh_grid.xdmf", "w")

    with io.XDMFFile(MPI.COMM_WORLD, './out_gmsh/mesh_grid.xdmf', 'r') as xdmf:
        # Read the mesh
        msh = xdmf.read_mesh(name="Grid")

    return x_coords, y_coords, msh
    
    
def visualize_mesh(V: dolfinx.fem.functionspace):
    """
    Creates a visualization of a mesh

    Args:
        V (dolfinx.fem.functionspace): The function space associated with the mesh
    """
    p = pyvista.Plotter()
    topology, cell_types, geometry = dolfinx.plot.vtk_mesh(V)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

    p.add_mesh(grid, style="wireframe", color="k")

    p.view_yx()
    p.show_axes()
    p.camera.roll -=90
    p.show()