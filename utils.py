import ase
import ase.io


def load_xyz(xyz_path):
    """Load xyz file into ase object

    Parameters
    ----------
    xyz_path : str
        path to xyz file

    Returns
    -------
    ase.Atoms
        ase object of xyz file
    """
    return ase.io.read(xyz_path)