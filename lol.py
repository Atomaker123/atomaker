import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QComboBox, QPushButton,
    QVBoxLayout, QHBoxLayout, QLabel, QFrame
)
from PyQt5.QtCore import Qt, QPropertyAnimation, QEasingCurve, QTimer
import numpy as np
from vispy import scene, app
from vispy.visuals import transforms
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtGui import QFont
from math import cos, sin, pi
from vispy.color import Color
import random
import json
import os

# Load element info from external JSON file if available (robust search)
ELEMENT_INFO = {}
_json_path_candidates = []
try:
    _json_path_candidates.append(os.path.join(os.path.dirname(__file__), 'elements.json'))
except Exception:
    pass
_json_path_candidates.append(os.path.join(os.getcwd(), 'elements.json'))
for _json_path in _json_path_candidates:
    if os.path.exists(_json_path):
        try:
            with open(_json_path, 'r', encoding='utf-8') as f:
                ELEMENT_INFO = json.load(f)
            break
        except Exception as e:
            print('Failed to load elements.json from', _json_path, ':', e)
            ELEMENT_INFO = {}

qt_app = QtWidgets.QApplication(sys.argv)

# Determine a DPI scale factor so UI element sizes remain consistent
# across devices with different pixel densities. Base DPI is assumed 96.
try:
    _screen = qt_app.primaryScreen()
    SCALE = float(_screen.logicalDotsPerInch()) / 96.0 if _screen is not None else 1.0
except Exception:
    SCALE = 1.0
# Ensure VisPy uses the PyQt5 backend to integrate with the Qt event loop
try:
    app.use_app('pyqt5')
except Exception:
    pass

canvas = scene.SceneCanvas(keys='interactive', size=(int(1000 * SCALE), int(600 * SCALE)), show=False)
canvas.bgcolor = 'white'
canvas.native.setFocusPolicy(QtCore.Qt.NoFocus)

view = canvas.central_widget.add_view()
view.camera = scene.TurntableCamera(up='z', fov=60, distance=8)
view.camera.center = (0, 0, 0)

protons, neutrons, electrons = [], [], []
electron_params = []
electron_trails = []
electron_trail_points = []  # Store static trail points for each electron

black_hole_visual = None
is_black_hole = False  # Track black hole state

NEUTRON_EXPLODE_COUNT = 5  # Number of neutrons to animate outward per explosion

chance_percent = 0.01

# Global element name mapping (used for AtomName and info sidebar)
ELEMENT_NAMES = {
    1: "Hydrogen",
    2: "Helium",
    3: "Lithium",
    4: "Beryllium",
    5: "Boron",
    6: "Carbon",
    7: "Nitrogen",
    8: "Oxygen",
    9: "Fluorine",
    10: "Neon",
    11: "Sodium",
    12: "Magnesium",
    13: "Aluminum",
    14: "Silicon",
    15: "Phosphorus",
    16: "Sulfur",
    17: "Chlorine",
    18: "Argon",
    19: "Potassium",
    20: "Calcium",
    21: "Scandium",
    22: "Titanium",
    23: "Vanadium",
    24: "Chromium",
    25: "Manganese",
    26: "Iron",
    27: "Cobalt",
    28: "Nickel",
    29: "Copper",
    30: "Zinc",
    31: "Gallium",
    32: "Germanium",
    33: "Arsenic",
    34: "Selenium",
    35: "Bromine",
    36: "Krypton",
    37: "Rubidium",
    38: "Strontium",
    39: "Yttrium",
    40: "Zirconium",
    41: "Niobium",
    42: "Molybdenum",
    43: "Technetium",
    44: "Ruthenium",
    45: "Rhodium",
    46: "Palladium",
    47: "Silver",
    48: "Cadmium",
    49: "Indium",
    50: "Tin",
    51: "Antimony",
    52: "Tellurium",
    53: "Iodine",
    54: "Xenon",
    55: "Cesium",
    56: "Barium",
    57: "Lanthanum",
    58: "Cerium",
    59: "Praseodymium",
    60: "Neodymium",
    61: "Promethium",
    62: "Samarium",
    63: "Europium",
    64: "Gadolinium",
    65: "Terbium",
    66: "Dysprosium",
    67: "Holmium",
    68: "Erbium",
    69: "Thulium",
    70: "Ytterbium",
    71: "Lutetium",
    72: "Hafnium",
    73: "Tantalum",
    74: "Tungsten",
    75: "Rhenium",
    76: "Osmium",
    77: "Iridium",
    78: "Platinum",
    79: "Gold",
    80: "Mercury",
    81: "Thallium",
    82: "Lead",
    83: "Bismuth",
    84: "Polonium",
    85: "Astatine",
    86: "Radon",
    87: "Francium",
    88: "Radium",
    89: "Actinium",
    90: "Thorium",
    91: "Protactinium",
    92: "Uranium",
    93: "Neptunium",
    94: "Plutonium",
    95: "Americium",
    96: "Curium",
    97: "Berkelium",
    98: "Californium",
    99: "Einsteinium",
    100: "Fermium",
    101: "Mendelevium",
    102: "Nobelium",
    103: "Lawrencium",
    104: "Rutherfordium",
    105: "Dubnium",
    106: "Seaborgium",
    107: "Bohrium",
    108: "Hassium",
    109: "Meitnerium",
    110: "Darmstadtium",
    111: "Roentgenium",
    112: "Copernicium",
    113: "Nihonium",
    114: "Flerovium",
    115: "Moscovium",
    116: "Livermorium",
    117: "Tennessine",
    118: "Oganesson",
    119: "Ununennium",
    120: "Unbinilium",
    121: "Unbiunium",
    122: "Unbibium ",
}

# Load element info from external JSON file if available
ELEMENT_INFO = {}
_json_path = os.path.join(os.path.dirname(__file__), 'elements.json')
if os.path.exists(_json_path):
    try:
        with open(_json_path, 'r', encoding='utf-8') as f:
            ELEMENT_INFO = json.load(f)
    except Exception as e:
        print('Failed to load elements.json:', e)

def random_nucleus_pos(scale=0.1):
    return np.random.normal(scale=scale, size=3)



def random_unit_vector():
    """Generate a random 3D unit vector."""
    phi = np.random.uniform(0, 2 * np.pi)
    costheta = np.random.uniform(-1, 1)
    sintheta = np.sqrt(1 - costheta**2)
    return np.array([
        sintheta * np.cos(phi),
        sintheta * np.sin(phi),
        costheta
    ])

def s_orbital(angle, radius=1.0, normal=None, offset_scale=0.7):
    """
    3D s-orbital: circle in a random plane defined by 'normal',
    always centered on the nucleus (no offset).
    """
    if normal is None:
        normal = np.array([0, 0, 1])
    normal = normal / np.linalg.norm(normal)
    # Find two orthogonal vectors in the plane
    if abs(normal[2]) < 0.99:
        v = np.cross(normal, [0, 0, 1])
    else:
        v = np.cross(normal, [0, 1, 0])
    v = v / np.linalg.norm(v)
    w = np.cross(normal, v)
    # Parametric circle in the plane
    pos = radius * (np.cos(angle) * v + np.sin(angle) * w)
    return pos  # No offset, always centered

def p_orbital(angle, radius=1.5, axis='x', offset_scale=0.25):
    t = angle
    a = radius * 0.7
    offset = np.array([radius * 0.3, radius * 0.3, radius * 0.3])
    if axis == 'x':
        x = radius * cos(t)
        y = a * sin(t)
        z = a * sin(2 * t)
        axis_vec = np.array([1,0,0])
    elif axis == 'y':
        x = a * sin(t)
        y = radius * cos(t)
        z = a * sin(2 * t)
        axis_vec = np.array([0,1,0])
    else:
        x = a * sin(t)
        y = a * sin(2 * t)
        z = radius * cos(t)
        axis_vec = np.array([0,0,1])
    # Add offset along axis normal
    axis_vec = axis_vec / np.linalg.norm(axis_vec)
    offset2 = axis_vec * (radius * offset_scale)
    return np.array([x, y, z]) + offset + offset2

def d_orbital(angle, radius=2.0, type_id=0, offset_scale=0.18):
    t = angle
    r = radius
    if type_id == 0:
        x = r * cos(t) * sin(t)
        y = r * sin(t) * sin(t)
        z = 0
        axis_vec = np.array([0,0,1])
    elif type_id == 1:
        x = r * (cos(t)**2 - sin(t)**2)
        y = r * 2 * sin(t) * cos(t)
        z = 0
        axis_vec = np.array([0,0,1])
    elif type_id == 2:
        x = r * cos(t)
        y = r * sin(t)
        z = r * cos(2*t)/2
        axis_vec = np.array([1,1,1])
    elif type_id == 3:
        x = r * cos(t)
        y = 0
        z = r * sin(t)
        axis_vec = np.array([0,1,0])
    else:
        x = 0
        y = r * cos(t)
        z = r * sin(t)
        axis_vec = np.array([1,0,0])
    axis_vec = axis_vec / np.linalg.norm(axis_vec)
    offset = axis_vec * (radius * offset_scale)
    return np.array([x, y, z]) + offset

def f_orbital(angle, radius=2.5, type_id=0, offset_scale=0.12):
    t = angle
    r = radius
    if type_id == 0:
        x = r * np.sin(3*t) * np.cos(t)
        y = r * np.sin(3*t) * np.sin(t)
        z = r * np.cos(3*t)
        axis_vec = np.array([1,1,1])
    else:
        x = r * np.cos(3*t) * np.cos(t)
        y = r * np.cos(3*t) * np.sin(t)
        z = r * np.sin(3*t)
        axis_vec = np.array([1,0,1])
    axis_vec = axis_vec / np.linalg.norm(axis_vec)
    offset = axis_vec * (radius * offset_scale)
    return np.array([x, y, z]) + offset

def assign_electron_orbital(idx):
    if idx == 0:
        # Only the very first electron gets an s-orbital
        normal = random_unit_vector()
        return ('s', np.random.uniform(0, 2*pi), normal)
    elif idx < 6:
        axis_map = ['x', 'x', 'y', 'y', 'z']
        return ('p', np.random.uniform(0, 2*pi), axis_map[idx - 1])
    elif idx < 16:
        type_id = (idx - 6) // 2
        return ('d', np.random.uniform(0, 2*pi), type_id)
    elif idx < 30:
        type_id = (idx - 16) % 2
        return ('f', np.random.uniform(0, 2*pi), type_id)
    else:
        # For higher electrons, assign random orbital type and orientation to avoid overlap
        orbital_types = ['p', 'd', 'f']
        orbital = np.random.choice(orbital_types)
        angle = np.random.uniform(0, 2*pi)
        if orbital == 'p':
            axis = np.random.choice(['x', 'y', 'z'])
            return ('p', angle, axis)
        elif orbital == 'd':
            type_id = np.random.randint(0, 5)
            return ('d', angle, type_id)
        else:
            type_id = np.random.randint(0, 2)
            return ('f', angle, type_id)

def get_nucleus_radius():
    all_positions = []
    for p in protons + neutrons:
        pos = p.transform.translate
        all_positions.append(pos)
    if not all_positions:
        return 0.2  # default small nucleus radius if none
    max_dist = max(np.linalg.norm(pos) for pos in all_positions)
    return max_dist + 0.12  # plus particle radius margin

def get_nucleus_center():
    all_positions = []
    for p in protons + neutrons:
        pos = np.array(p.transform.translate)
        if pos.shape[0] > 3:   # remove extra dimension if present
            pos = pos[:3]
        all_positions.append(pos)
    if not all_positions:
        return np.zeros(3)
    return np.mean(all_positions, axis=0)

def electron_position(params, nucleus_radius):
    orbital, angle, orientation = params
    base_radii = {'s': 1.5, 'p': 2.2, 'd': 2.7, 'f': 3.2}
    radius = base_radii.get(orbital, 1.5) + nucleus_radius + 0.3
    com = get_nucleus_center()
    if orbital == 's':
        normal = orientation
        pos = s_orbital(angle, radius, normal=normal, offset_scale=0.7)
    elif orbital == 'p':
        pos = p_orbital(angle, radius, axis=orientation, offset_scale=0.25)
    elif orbital == 'd':
        pos = d_orbital(angle, radius, type_id=orientation, offset_scale=0.18)
    elif orbital == 'f':
        pos = f_orbital(angle, radius, type_id=orientation, offset_scale=0.12)
    else:
        pos = np.array([0,0,0])
    return pos + com  # offset by nucleus COM




def is_position_free(pos, existing_positions, min_distance=0.3):
    """Return True if 'pos' is at least min_distance away from all existing_positions."""
    for p in existing_positions:
        p = np.array(p)
        if p.shape[0] > 3:
            p = p[:3]
        if np.linalg.norm(pos - p) < min_distance:
            return False
    return True
def add_particle(particle_type):
    global window
    global is_black_hole
    if is_black_hole:
        return
    if particle_type in ('proton', 'neutron'):
        color = 'red' if particle_type == 'proton' else 'yellow'
        radius = 0.12
        max_attempts = 100

        existing_positions = []
        for p in protons + neutrons:
            existing_positions.append(p.transform.translate)

        for _ in range(max_attempts):
            pos = random_nucleus_pos()
            if is_position_free(pos, existing_positions, min_distance=2*radius + 0.05):
                break
        else:
            # If can't find free spot, just use last position (may overlap)
            pass

        sphere = scene.visuals.Sphere(
            radius=radius, method='latitude', color=color,
            edge_color=color, shading='smooth', rows=20, cols=20  # reduced mesh complexity
        )
        sphere.transform = transforms.STTransform(translate=pos)
        view.add(sphere)
        if particle_type == 'proton':
            protons.append(sphere)
        else:
            neutrons.append(sphere)

    elif particle_type == 'electron':
        idx = len(electrons)
        color = 'blue'
        radius = 0.06  # electron sphere size

        nucleus_radius = get_nucleus_radius()
        nucleus_center = get_nucleus_center()  # Anchor for trail
        params = assign_electron_orbital(idx)
        electron_params.append(params)

        pos = electron_position(params, nucleus_radius)

        orbital, angle, orientation = params
        trail_points = []
        trail_steps = 100  # increased from 40 for smooth orbits
        base_radii = {'s': 1.5, 'p': 2.2, 'd': 2.7, 'f': 3.2}
        trail_radius = base_radii.get(orbital, 1.5) + nucleus_radius + 0.3  # Use a separate variable
        for i in range(trail_steps):
            trail_angle = 2 * pi * i / trail_steps
            trail_params = (orbital, trail_angle, orientation)
            if orbital == 's':
                normal = orientation
                trail_pos = s_orbital(trail_angle, trail_radius, normal=normal, offset_scale=0.7)
            elif orbital == 'p':
                trail_pos = p_orbital(trail_angle, trail_radius, axis=orientation, offset_scale=0.25)
            elif orbital == 'd':
                trail_pos = d_orbital(trail_angle, trail_radius, type_id=orientation, offset_scale=0.18)
            elif orbital == 'f':
                trail_pos = f_orbital(trail_angle, trail_radius, type_id=orientation, offset_scale=0.12)
            else:
                trail_pos = np.array([0,0,0])
            trail_points.append(trail_pos + nucleus_center)
        electron_trail_points.append(trail_points)

        trail = scene.visuals.Line(
            np.array(trail_points), color=(0.2, 0.7, 1.0, 0.7),
            width=1, method='gl'
        )
        view.add(trail)
        electron_trails.append(trail)

        sphere = scene.visuals.Sphere(
            radius=radius, method='latitude', color=color,
            edge_color=color, shading='smooth', rows=20, cols=20  # reduced mesh complexity
        )
        sphere.transform = transforms.STTransform(translate=pos)
        view.add(sphere)
        electrons.append(sphere)

    canvas.update()
    if 'window' in globals() and window:
        window.update_counter()
    # Black hole check after every addition
    p = len(protons)
    n = len(neutrons)
    if abs(n - p) >= 150:
        make_black_hole()
        return
    # If large imbalance (>10) schedule a decay for the species in excess
    electron_excess = len(electrons) - p
    diff = n - p
    if diff > 10 or diff < -10 or electron_excess > 10:
        if 'window' in globals() and window:
            # determine which species is in excess
            if diff > 10:
                species = 'neutron'
            elif diff < -10:
                species = 'proton'
            else:
                species = 'electron'
            window.show_decay_timer(10, species)
        def start_decay(sp=species):
            if 'window' in globals() and window:
                window.hide_decay_timer()
            if sp == 'neutron':
                animate_neutron_decay(target_diff=3)
            elif sp == 'proton':
                animate_proton_decay(target_diff=3)
            else:
                animate_electron_decay(target_diff=3)
        QTimer.singleShot(10000, start_decay)
    print(f"Added {particle_type}")

def update_electrons(ev):
    speed = 0.03
    nucleus_radius = get_nucleus_radius()
    nucleus_center = get_nucleus_center()
    for i in range(len(electrons)):
        orbital, angle, orientation = electron_params[i]
        angle += speed
        electron_params[i] = (orbital, angle, orientation)
        # Use the current nucleus center for electron position
        pos = electron_position(electron_params[i], nucleus_radius)
        electrons[i].transform.translate = pos
        # Do NOT update electron_trails[i] data; trails remain static
    canvas.update()

def AtomName():
    global is_black_hole
    if is_black_hole:
        return "BLACK HOLE"
    p = len(protons)
    n = len(neutrons)
    e = len(electrons)

    # If nothing is present, show nothing
    if p == 0 and n == 0 and e == 0:
        return ""

    # Black hole check FIRST
    if abs(n - p) >= 150:
        make_black_hole()
        return "BLACK HOLE"

    base_name = ELEMENT_NAMES.get(p, f"Element Z={p}")

    # Require at least 1 proton and 1 electron for stability
    if p < 1 or e < 1:
        return f"{base_name} (Unstable)"

    # Stability check: protons vs neutrons and protons vs electrons
    if abs(p - n) > 2 or abs(p - e) > 5:
        return f"{base_name} (Unstable)"
    else:
        return f"{base_name} (Stable)"
    
        

def make_black_hole():
    global black_hole_visual, is_black_hole
    is_black_hole = True
    # Remove all protons, neutrons, electrons, and their visuals
    for p in protons:
        p.parent = None
    protons.clear()
    for n in neutrons:
        n.parent = None
    neutrons.clear()
    for e in electrons:
        e.parent = None
    electrons.clear()
    for t in electron_trails:
        t.parent = None
    electron_trails.clear()
    electron_params.clear()
    electron_trail_points.clear()
    canvas.update()
    # Remove any existing black hole visual
    if black_hole_visual is not None:
        black_hole_visual.parent = None
        black_hole_visual = None
    # Add a large black sphere at the center
    black_hole_visual = scene.visuals.Sphere(
        radius=0.7, method='latitude', color='black',
        edge_color='black', shading='smooth', rows=32, cols=32
    )
    black_hole_visual.transform = transforms.STTransform(translate=(0, 0, 0))
    view.add(black_hole_visual)
    # Update the label
    if 'window' in globals() and window:
        window.canvas_label.setText("BLACK HOLE")

def reset_atom():
    global is_black_hole, black_hole_visual
    # Remove all visuals
    for p in protons:
        p.parent = None
    protons.clear()
    for n in neutrons:
        n.parent = None
    neutrons.clear()
    for e in electrons:
        e.parent = None
    electrons.clear()
    for t in electron_trails:
        t.parent = None
    electron_trails.clear()
    electron_params.clear()
    electron_trail_points.clear()
    if black_hole_visual is not None:
        black_hole_visual.parent = None
        black_hole_visual = None
    is_black_hole = False
    canvas.update()
    # Restart electron animation timer if needed
    if 'window' in globals() and hasattr(window, 'electron_timer'):
        if not window.electron_timer.isActive():
            window.electron_timer.start(16)
    if 'window' in globals() and window:
        window.update_counter()
        window.update_atom_name()

def explode():
    # Animate all particles outward, then remove them
    duration = 3500  # ms (longer explosion)
    steps = 90      # smoother animation
    interval = duration // steps
    # Stop electron animation timer to prevent glitches
    if 'window' in globals() and hasattr(window, 'electron_timer'):
        window.electron_timer.stop()
    directions = []
    visuals = protons + neutrons + electrons
    for _ in visuals:
        dir = np.random.normal(size=3)
        dir = dir / np.linalg.norm(dir)
        directions.append(dir)
    positions = []
    for v in visuals:
        pos = np.array(v.transform.translate)
        if pos.shape[0] > 3:
            pos = pos[:3]
        positions.append(pos)
    def animate(step=0):
        if step >= steps:
            # Remove all visuals from scene first
            for v in visuals:
                if v.parent is not None:
                    v.parent = None
            for t in electron_trails:
                if t.parent is not None:
                    t.parent = None
            # Now clear all lists
            protons.clear()
            neutrons.clear()
            electrons.clear()
            electron_params.clear()
            electron_trail_points.clear()
            electron_trails.clear()
            canvas.update()
            if 'window' in globals() and window:
                window.update_counter()
                window.update_atom_name()
            return
        for i, v in enumerate(visuals):
            if v.parent is not None:
                v.transform.translate = positions[i] + directions[i] * (step / steps) * 4.0
        canvas.update()
        QTimer.singleShot(interval, lambda: animate(step+1))
    animate()

def neutron_explode():
    # Animate a set number of neutrons outward, then remove them
    duration = 1200  # ms
    steps = 300
    interval = duration // steps
    if len(neutrons) == 0:
        return
    count = min(NEUTRON_EXPLODE_COUNT, len(neutrons))
    selected = np.random.choice(neutrons, size=count, replace=False)
    visuals = list(selected)
    directions = []
    for _ in visuals:
        dir = np.random.normal(size=3)
        dir = dir / np.linalg.norm(dir)
        directions.append(dir)
    positions = []
    for v in visuals:
        pos = np.array(v.transform.translate)
        if pos.shape[0] > 3:
            pos = pos[:3]
        positions.append(pos)

    # --- Freeze nucleus center/radius for electrons during animation ---
    cached_center = get_nucleus_center()
    cached_radius = get_nucleus_radius()
    orig_get_nucleus_center = globals()['get_nucleus_center']
    orig_get_nucleus_radius = globals()['get_nucleus_radius']
    def frozen_center():
        return cached_center
    def frozen_radius():
        return cached_radius
    globals()['get_nucleus_center'] = frozen_center
    globals()['get_nucleus_radius'] = frozen_radius

    def animate(step=0):
        if step >= steps:
            for v in visuals:
                v.parent = None
            for v in visuals:
                if v in neutrons:
                    neutrons.remove(v)
            # Restore nucleus center/radius functions
            globals()['get_nucleus_center'] = orig_get_nucleus_center
            globals()['get_nucleus_radius'] = orig_get_nucleus_radius
            canvas.update()
            if 'window' in globals() and window:
                window.update_counter()
                window.update_atom_name()
            return
        for i, v in enumerate(visuals):
            if v.parent is not None:
                v.transform.translate = positions[i] + directions[i] * (step / steps) * 4.0
        canvas.update()
        QTimer.singleShot(interval, lambda: animate(step+1))
    animate()

def animate_neutron_decay(target_diff=3):
    # Animate removal of neutrons until |n-p| <= target_diff
    def decay_step():
        p = len(protons)
        n = len(neutrons)
        if abs(n - p) <= target_diff or not neutrons:
            if 'window' in globals() and window:
                window.update_counter()
                window.update_atom_name()
            return
        # Animate one neutron outward
        v = neutrons.pop()
        pos = np.array(v.transform.translate)
        if pos.shape[0] > 3:
            pos = pos[:3]
        dir = np.random.normal(size=3)
        dir = dir / np.linalg.norm(dir)
        steps = 30
        interval = 30
        def animate(step=0):
            if step >= steps:
                v.parent = None
                canvas.update()
                QTimer.singleShot(50, decay_step)  # Continue decay after this neutron
                return
            v.transform.translate = pos + dir * (step / steps) * 4.0
            canvas.update()
            QTimer.singleShot(interval, lambda: animate(step+1))
        animate()
    decay_step()


def animate_proton_decay(target_diff=3):
    # Animate removal of protons until p - n <= target_diff
    def decay_step():
        p = len(protons)
        n = len(neutrons)
        if p - n <= target_diff or not protons:
            if 'window' in globals() and window:
                window.update_counter()
                window.update_atom_name()
            return
        # Animate one proton outward
        v = protons.pop()
        pos = np.array(v.transform.translate)
        if pos.shape[0] > 3:
            pos = pos[:3]
        dir = np.random.normal(size=3)
        dir = dir / np.linalg.norm(dir)
        steps = 30
        interval = 30
        def animate(step=0):
            if step >= steps:
                v.parent = None
                canvas.update()
                QTimer.singleShot(50, decay_step)
                return
            v.transform.translate = pos + dir * (step / steps) * 4.0
            canvas.update()
            QTimer.singleShot(interval, lambda: animate(step+1))
        animate()
    decay_step()


def animate_electron_decay(target_diff=3):
    # Animate removal of electrons until e - p <= target_diff
    def decay_step():
        p = len(protons)
        e = len(electrons)
        if e - p <= target_diff or not electrons:
            if 'window' in globals() and window:
                window.update_counter()
                window.update_atom_name()
            return
        # Animate one electron outward (also remove its trail and params)
        v = electrons.pop()
        # Also remove corresponding trail and params if present
        trail = None
        if electron_trails:
            try:
                trail = electron_trails.pop()
            except Exception:
                trail = None
        if electron_params:
            try:
                electron_params.pop()
            except Exception:
                pass
        if electron_trail_points:
            try:
                electron_trail_points.pop()
            except Exception:
                pass
        pos = np.array(v.transform.translate)
        if pos.shape[0] > 3:
            pos = pos[:3]
        dir = np.random.normal(size=3)
        dir = dir / np.linalg.norm(dir)
        steps = 30
        interval = 30
        def animate(step=0):
            if step >= steps:
                v.parent = None
                if trail is not None:
                    trail.parent = None
                canvas.update()
                QTimer.singleShot(50, decay_step)
                return
            v.transform.translate = pos + dir * (step / steps) * 4.0
            canvas.update()
            QTimer.singleShot(interval, lambda: animate(step+1))
        animate()
    decay_step()
# ...existing code...

class AnimatedSidebar(QWidget):
    def add_protons(self, amount=1):
        if globals().get('is_black_hole', False):
            return
        for _ in range(amount):
            add_particle(particle_type='proton')
        self.update_atom_name()
        
    def update_atom_name(self):
        self.canvas_label.setText(AtomName())

    def add_electrons(self, amount=1):
        if globals().get('is_black_hole', False):
            return
        for _ in range(amount):
            add_particle(particle_type='electron')
        self.update_atom_name()

    def add_neutrons(self, amount=1):
        if globals().get('is_black_hole', False):
            return
        for _ in range(amount):
            add_particle(particle_type='neutron')
        self.update_atom_name()

    def remove_protons(self, amount=1):
        if globals().get('is_black_hole', False):
            return
        for _ in range(amount):
            if protons:
                p = protons.pop()
                p.parent = None  # Correct way to remove from scene
                canvas.update()
        self.update_counter()
        self.update_atom_name()

    def remove_electrons(self, amount=1):
        if globals().get('is_black_hole', False):
            return
        for _ in range(amount):
            if electrons:
                e = electrons.pop()
                e.parent = None  # Correct way to remove from scene
                if electron_trails:
                    t = electron_trails.pop()
                    t.parent = None  # Correct way to remove from scene
                if electron_params:
                    electron_params.pop()
                if electron_trail_points:
                    electron_trail_points.pop()
                canvas.update()
        self.update_counter()
        self.update_atom_name()

    def remove_neutrons(self, amount=1):
        if globals().get('is_black_hole', False):
            return
        for _ in range(amount):
            if neutrons:
                n = neutrons.pop()
                n.parent = None  # Correct way to remove from scene
                canvas.update()
        self.update_counter()
        self.update_atom_name()

    def update_counter(self):
        text = f"Protons: {len(protons)} | Neutrons: {len(neutrons)} | Electrons: {len(electrons)}"
        self.counter_label.setText(text)
        self.update_atom_name()  # Always update label and black hole logic after any change
        # Also update info panel
        self.update_info_panel()

    def update_info_panel(self):
        # Build formatted info text from ELEMENT_INFO JSON with headings and new fields
        p = len(protons)
        n = len(neutrons)
        e = len(electrons)
        if p < 1:
            try:
                self.info_label_right.setText("No nucleus present.")
            except Exception:
                pass
            return
        elem = ELEMENT_INFO.get(str(p)) if isinstance(ELEMENT_INFO, dict) else None
        base_name = ELEMENT_NAMES.get(p, f"Element Z={p}")
        stability = "Stable" if (abs(p - n) <= 2 and abs(p - e) <=5 and p>=1 and e>=1) else "Unstable"
        mass_number = p + n
        
        # Format with rich text markup for headings
        info_lines = [
            f"<b>Atomic Number:</b> {p}",
            f"<b>Neutrons:</b> {n}",
            f"<b>Electrons:</b> {e}",
            f"<b>Mass Number:</b> {mass_number}",
            f"<b>Stability:</b> {stability}"
        ]
        
        # Isotope info
        isotope_text = None
        if elem and 'isotopes' in elem and isinstance(elem['isotopes'], dict):
            iso_entry = elem['isotopes'].get(str(mass_number)) or elem['isotopes'].get(str(n))
            if iso_entry:
                name = iso_entry.get('name') or f"Isotope {mass_number}"
                info = iso_entry.get('info') or ''
                isotope_text = f"<b>Isotope:</b> {name}<br>{info}" if info else f"<b>Isotope:</b> {name}"
                if 'uses' in iso_entry:
                    isotope_text += f"<br><b>Uses:</b> {iso_entry['uses']}"
                if 'natural_occurrence' in iso_entry:
                    isotope_text += f"<br><b>Natural Occurrence:</b> {iso_entry['natural_occurrence']}"
        if isotope_text is None and n != p:
            isotope_text = f"<b>Isotope-like:</b> mass number {mass_number} (N={n})"
        if isotope_text:
            info_lines.append(isotope_text)
        
        # Element-level info
        if elem:
            if 'description' in elem:
                info_lines.append(f"<b>Description:</b> {elem['description']}")
            if 'uses' in elem:
                info_lines.append(f"<b>Uses:</b> {elem['uses']}")
            if 'natural_occurrence' in elem:
                info_lines.append(f"<b>Natural Occurrence:</b> {elem['natural_occurrence']}")
            if 'history' in elem:
                info_lines.append(f"<b>History:</b> {elem['history']}")
        
        text = '<br><br>'.join(info_lines)
        try:
            self.info_label_right.setText(text)
        except Exception:
            try:
                self.info_label.setText(text)
            except Exception:
                pass

    def add_reset_button(self):
        reset_btn = QPushButton("RESET")
        reset_btn.setStyleSheet("""
            QPushButton {
                background-color: #ff9800;
                color: white;
                font-weight: bold;
                border-radius: 5px;
                padding: 10px 20px;
                font-size: 16px;
                margin-top: 20px;
            }
            QPushButton:hover {
                background-color: #e65100;
            }
        """)
        reset_btn.clicked.connect(reset_atom)
        self.sidebar_layout.addWidget(reset_btn)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Animated Sidebar Fullscreen")
        self.sidebar_expanded = True
        # Use DPI-scaled stable widths so layout looks the same on all displays
        # Reduce default width to be less excessive while keeping text readable
        self.sidebar_width = int(240 * SCALE)
        self.init_ui()
        self.counter_label = QLabel()
        # Improved counter label styling for readability
        self.counter_label.setStyleSheet("color: white; font-weight: bold; margin-top: 15px; font-size: 14px; padding:4px;")
        self.counter_label.setWordWrap(True)
        self.sidebar_layout.addWidget(self.counter_label)
        # self.add_explode_button()  # Removed EXPLODE button
        # self.add_neutron_explode_button()  # Removed NEUTRON EXPLODE button
        self.add_reset_button()    # Add RESET button
        self.showFullScreen()
        # Start electron animation timer
        self.electron_timer = QTimer(self)
        self.electron_timer.timeout.connect(lambda: update_electrons(None))
        self.electron_timer.start(16)  # ~60 FPS

        # Start the background task for atom
        self.run_atom_background_task()

    def init_ui(self):
        self.main_layout = QHBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        # Larger, visible Close and Toggle buttons (modern gradient look)
        self.close_btn = QPushButton("✕")
        self.close_btn.setFixedSize(int(36 * SCALE), int(36 * SCALE))
        self.close_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #ff6b6b, stop:1 #c62828);
                color: white;
                font-weight: bold;
                font-size: 18px;
                border-radius: 8px;
                border: 1px solid rgba(0,0,0,0.25);
                padding: 4px;
            }
            QPushButton:hover { background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #ff5252, stop:1 #b71c1c); }
        """)
        self.close_btn.clicked.connect(QApplication.quit)

        self.toggle_btn = QPushButton("☰")
        self.toggle_btn.setFixedSize(int(36 * SCALE), int(36 * SCALE))
        self.toggle_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #ff8a65, stop:1 #d45353);
                color: white;
                font-weight: bold;
                font-size: 18px;
                border-radius: 8px;
                border: 1px solid rgba(0,0,0,0.18);
            }
            QPushButton:hover { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #ff7043, stop:1 #c03939); }
        """)
        # Toggle the left sidebar
        self.toggle_btn.clicked.connect(self.toggle_sidebar)

        # Fullscreen toggle button (modern)
        self.fullscreen_btn = QPushButton("⛶")
        self.fullscreen_btn.setFixedSize(int(36 * SCALE), int(36 * SCALE))
        self.fullscreen_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #66bb6a, stop:1 #43a047);
                color: white;
                border-radius: 8px;
                border: 1px solid rgba(0,0,0,0.14);
            }
            QPushButton:hover { background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #57a05a, stop:1 #2e7d32); }
        """)
        self.fullscreen_btn.clicked.connect(self.toggle_fullscreen)

        # Theme toggle button (modern)
        self.theme_btn = QPushButton("☼")
        self.theme_btn.setFixedSize(int(36 * SCALE), int(36 * SCALE))
        self.theme_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #ffd54f, stop:1 #ffb300);
                color: #222;
                border-radius: 8px;
                border: 1px solid rgba(0,0,0,0.12);
            }
            QPushButton:hover { background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #ffca28, stop:1 #ffb300); }
        """)
        self.theme_btn.clicked.connect(self.toggle_theme)

        # Info panel toggle button (modern)
        self.info_btn = QPushButton("ℹ")
        self.info_btn.setFixedSize(int(36 * SCALE), int(36 * SCALE))
        self.info_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #7c9ed9, stop:1 #4a5f9f);
                color: white;
                font-weight: bold;
                font-size: 20px;
                border-radius: 8px;
                border: 1px solid rgba(0,0,0,0.16);
            }
            QPushButton:hover { background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #5a7db5, stop:1 #2d3b5f); }
        """)
        self.info_btn.clicked.connect(self.toggle_info_panel)

        toggle_layout = QVBoxLayout()
        toggle_layout.setContentsMargins(int(8 * SCALE), int(8 * SCALE), int(8 * SCALE), int(8 * SCALE))
        toggle_layout.setSpacing(int(10 * SCALE))
        # Add a top stretch so the toggle buttons stay centered vertically when resizing
        toggle_layout.addWidget(self.close_btn)
        toggle_layout.addWidget(self.toggle_btn)
        toggle_layout.addWidget(self.fullscreen_btn)
        toggle_layout.addWidget(self.theme_btn)
        toggle_layout.addWidget(self.info_btn)
        toggle_layout.addStretch()

        toggle_widget = QWidget()
        toggle_widget.setLayout(toggle_layout)
        toggle_widget.setFixedWidth(int(64 * SCALE))
        toggle_widget.setStyleSheet("""
            QWidget { background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #0f1724, stop:1 #1f2430); border-right: 1px solid rgba(255,255,255,0.03); }
        """)

        self.sidebar = QFrame()
        # Keep the sidebar at a stable width so it doesn't get cramped, but allow animation by using max/min
        # Keep a fixed width for sidebar so it doesn't visually compress on different screens
        self.sidebar.setMaximumWidth(self.sidebar_width)
        self.sidebar.setMinimumWidth(self.sidebar_width)
        # Modern subtle gradient sidebar styling
        self.sidebar.setStyleSheet("""
            QFrame { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #15171a, stop:1 #24262a); border-right: 1px solid rgba(255,255,255,0.04); }
            QLabel { font-family: 'Segoe UI', 'Verdana', 'Arial'; color: #e6eef6; }
        """)
        self.sidebar_layout = QVBoxLayout(self.sidebar)
        # Align content to top with generous spacing
        self.sidebar_layout.setContentsMargins(int(24 * SCALE), int(16 * SCALE), int(24 * SCALE), int(16 * SCALE))
        self.sidebar_layout.setSpacing(int(18 * SCALE))
        self.sidebar_layout.setAlignment(Qt.AlignTop)

        # Sidebar content will remain controls only; info panel will be on the right
        # (Removed early addStretch to avoid large top gap)
        # Track sidebar buttons and labels for dynamic height/font scaling on resize
        self._sidebar_buttons = []
        self._sidebar_labels = []

        def create_button_row(label_text, add_callback, remove_callback):
            label = QLabel(label_text)
            label.setStyleSheet("color: #e6eef6; font-weight: 700;")
            # Use a font object so the resize handler can scale it (base 16pt)
            lf = label.font()
            lf.setPointSizeF(16.0)
            label.setFont(lf)
            self.sidebar_layout.addWidget(label)
            self._sidebar_labels.append(label)

            add_container = QWidget()
            add_layout = QHBoxLayout(add_container)
            add_layout.setContentsMargins(0, 0, 0, 0)
            add_layout.setSpacing(int(12 * SCALE))
            for text, cb in [("+1", add_callback), ("+10", add_callback), ("+50", add_callback)]:
                btn = QPushButton(text)
                btn.setFixedHeight(int(44 * SCALE))
                btn.setStyleSheet("""
                    QPushButton {
                        background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #1643a3, stop:1 #0b246e);
                        color: #ffffff;
                        border-radius: 10px;
                        padding: 8px 14px;
                        font-weight: 700;
                        font-size: 15px;
                        border: 1px solid rgba(0,0,0,0.18);
                    }
                    QPushButton:hover { background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #0e2f7a, stop:1 #071744); }
                """)
                # Set a base font size for scaling
                bf = btn.font()
                bf.setPointSizeF(13.0)
                btn.setFont(bf)
                amount = int(text.replace("+", ""))
                btn.clicked.connect(lambda checked, a=amount, cb=cb: cb(a))
                self._sidebar_buttons.append(btn)
                add_layout.addWidget(btn)
            self.sidebar_layout.addWidget(add_container)

            remove_container = QWidget()
            remove_layout = QHBoxLayout(remove_container)
            remove_layout.setContentsMargins(0, 0, 0, 0)
            remove_layout.setSpacing(int(12 * SCALE))
            for text, cb in [("-1", remove_callback), ("-10", remove_callback), ("-50", remove_callback)]:
                btn = QPushButton(text)
                btn.setFixedHeight(int(44 * SCALE))
                btn.setStyleSheet("""
                    QPushButton {
                        background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #6e0c0c, stop:1 #2b0000);
                        color: #ffffff;
                        border-radius: 10px;
                        padding: 8px 14px;
                        font-weight: 700;
                        font-size: 15px;
                        border: 1px solid rgba(0,0,0,0.22);
                    }
                    QPushButton:hover { background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #4d0a0a, stop:1 #1a0000); }
                """)
                bf = btn.font()
                bf.setPointSizeF(13.0)
                btn.setFont(bf)
                amount = int(text.replace("-", ""))
                btn.clicked.connect(lambda checked, a=amount, cb=cb: cb(a))
                self._sidebar_buttons.append(btn)
                remove_layout.addWidget(btn)
            self.sidebar_layout.addWidget(remove_container)

        create_button_row("PROTONS", self.add_protons, self.remove_protons)
        create_button_row("ELECTRONS", self.add_electrons, self.remove_electrons)
        create_button_row("NEUTRONS", self.add_neutrons, self.remove_neutrons)

        # Black Hole button removed

        self.content = QWidget()
        self.content.setStyleSheet("background-color: white;")
        content_layout = QVBoxLayout(self.content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)

        self.canvas_label = QLabel(AtomName())
        self.canvas_label.setAlignment(Qt.AlignCenter)
        self.canvas_label.setMinimumHeight(int(60 * SCALE))
        self.canvas_label.setMaximumHeight(int(90 * SCALE))
        # Modern polished label styling for the canvas title (glass gradient)
        self.canvas_label.setStyleSheet("""
            QLabel {
                font-size: 18px;
                font-weight: 700;
                padding: 12px 8px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 rgba(255,255,255,0.95), stop:1 rgba(240,246,255,0.95));
                color: #1f2937;
                border-bottom: 1px solid rgba(31,41,55,0.06);
            }
        """)
        content_layout.addWidget(self.canvas_label)
        # Create decay timer label (but don't show yet)
        self.decay_timer_label = QLabel()
        self.decay_timer_label.setAlignment(Qt.AlignCenter)
        # Modern decay timer styling
        self.decay_timer_label.setStyleSheet("""
            QLabel {
                font-size: 18px;
                color: #7f1d1d;
                font-weight: 700;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #fff1f0, stop:1 #ffe6e6);
                border: 1px solid #f3a5a5;
                border-radius: 10px;
                padding: 8px 12px;
                margin: 10px;
            }
        """)
        self.decay_timer_label.setMinimumWidth(int(220 * SCALE))
        self.decay_timer_label.hide()
        content_layout.addWidget(self.decay_timer_label)
        # ----------------------

        # Ensure the VisPy canvas has a reasonable minimum size (DPI-scaled)
        canvas.native.setMinimumSize(int(800 * SCALE), int(480 * SCALE))
        content_layout.addWidget(canvas.native)

        # Right-side info panel (subdued, toggleable, modern gradient)
        self.info_width = int(320 * SCALE)
        self.info_panel_right = QFrame()
        # Allow animation by setting a maximum width; start visible at info_width
        self.info_panel_right.setMinimumWidth(0)
        self.info_panel_right.setMaximumWidth(self.info_width)
        self.info_panel_right.setStyleSheet("""
            QFrame { background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #ffffff, stop:1 #f0f5ff); border-left: 1px solid rgba(15,23,42,0.06); border-radius:8px; }
            QLabel#InfoTitle { color: #0f1724; font-weight: 800; font-size: 15px; }
            QLabel#InfoText { color: #334155; font-size: 13px; }
        """)
        self.info_layout_right = QVBoxLayout(self.info_panel_right)
        self.info_layout_right.setContentsMargins(int(14 * SCALE), int(14 * SCALE), int(14 * SCALE), int(14 * SCALE))
        self.info_layout_right.setSpacing(int(10 * SCALE))
        # Info panel header with a close button on the right (shorter and broader)
        header_widget = QWidget()
        header_layout = QHBoxLayout(header_widget)
        header_layout.setContentsMargins(0, 4, 0, 4)
        header_layout.setSpacing(int(6 * SCALE))
        header_widget.setFixedHeight(int(38 * SCALE))
        self.info_title_right = QLabel("ELEMENT INFO")
        self.info_title_right.setObjectName('InfoTitle')
        header_layout.addWidget(self.info_title_right)
        header_layout.addStretch()
        self.info_close_btn = QPushButton("✕")
        self.info_close_btn.setFixedSize(int(28 * SCALE), int(28 * SCALE))
        self.info_close_btn.setStyleSheet("""
            QPushButton { background: transparent; color: #334155; border: none; font-weight: 800; }
            QPushButton:hover { color: #111827; }
        """)
        self.info_close_btn.clicked.connect(self.toggle_info_panel)
        header_layout.addWidget(self.info_close_btn)
        self.info_layout_right.addWidget(header_widget)
        self.info_label_right = QLabel("")
        self.info_label_right.setObjectName('InfoText')
        self.info_label_right.setWordWrap(True)
        self.info_label_right.setTextFormat(Qt.RichText)
        self.info_layout_right.addWidget(self.info_label_right)

        self.main_layout.addWidget(toggle_widget)
        self.main_layout.addWidget(self.sidebar)
        self.main_layout.addWidget(self.content)
        self.main_layout.addWidget(self.info_panel_right)

        # Animation to toggle the right info panel width
        self.info_animation = QPropertyAnimation(self.info_panel_right, b"maximumWidth")
        self.info_animation.setDuration(900)
        self.info_animation.setEasingCurve(QEasingCurve.InOutCubic)
        self.info_panel_visible = True

        # Ensure sidebar animation exists and connect finished signals
        self.animation = QPropertyAnimation(self.sidebar, b"maximumWidth")
        self.animation.setDuration(900)
        self.animation.setEasingCurve(QEasingCurve.InOutCubic)
        self.animation.finished.connect(self._on_sidebar_animation_finished)
        self.info_animation.finished.connect(self._on_info_animation_finished)
        # Track theme: start with light theme
        self._theme_dark = False

    def toggle_sidebar(self):
        # Animate sidebar open/closed and update minimum width when done to preserve layout
        if self.sidebar_expanded:
            self.animation.setStartValue(self.sidebar_width)
            self.animation.setEndValue(0)
        else:
            # Ensure the widget can grow before animating
            self.sidebar.setMinimumWidth(0)
            self.animation.setStartValue(0)
            self.animation.setEndValue(self.sidebar_width)
        self.animation.start()
        # flip expected state immediately so the finished handler can use it
        self.sidebar_expanded = not self.sidebar_expanded

    def _on_sidebar_animation_finished(self):
        # After animation, enforce a sensible minimum width when expanded, or allow 0 when collapsed
        if self.sidebar_expanded:
            self.sidebar.setMinimumWidth(self.sidebar_width)
        else:
            self.sidebar.setMinimumWidth(0)

    def toggle_info_panel(self):
        # Show/hide the right info panel with an animation
        if self.info_panel_visible:
            # collapse
            start_w = self.info_panel_right.width() or self.info_panel_right.maximumWidth() or 320
            self.info_animation.setStartValue(start_w)
            self.info_animation.setEndValue(0)
            self.info_animation.start()
            # will hide at end
        else:
            # expand
            self.info_panel_right.setVisible(True)
            self.info_animation.setStartValue(0)
            self.info_animation.setEndValue(320)
            self.info_animation.start()
        self.info_panel_visible = not self.info_panel_visible

    def _on_info_animation_finished(self):
        if not self.info_panel_visible:
            # fully collapsed -> hide to remove from layout and avoid tiny cramped space
            self.info_panel_right.setVisible(False)
        else:
            # restored -> ensure visible and max width set
            self.info_panel_right.setVisible(True)
            self.info_panel_right.setMaximumWidth(320)

    def toggle_fullscreen(self):
        if self.isFullScreen():
            self.setWindowState(self.windowState() & ~QtCore.Qt.WindowFullScreen)
        else:
            self.setWindowState(self.windowState() | QtCore.Qt.WindowFullScreen)

    def resizeEvent(self, event):
        # Scale certain widget heights and fonts proportionally to window height.
        # We only grow sizes (never shrink below base) to keep buttons usable.
        try:
            h = max(1, self.height())
            scale = max(1.0, float(h) / float(int(720 * SCALE)))
            base_toggle = int(44 * SCALE)
            base_btn_h = int(40 * SCALE)
            # Toggle buttons (square)
            new_toggle = int(max(base_toggle, round(base_toggle * scale)))
            for b in [self.close_btn, self.toggle_btn, self.fullscreen_btn, self.theme_btn, self.info_btn]:
                b.setFixedSize(new_toggle, new_toggle)
                f = b.font()
                f.setPointSizeF(max(11.0, 11.0 * scale))
                b.setFont(f)
            # Sidebar +/- buttons
            for b in getattr(self, '_sidebar_buttons', []):
                new_h = int(max(base_btn_h, round(base_btn_h * scale)))
                b.setFixedHeight(new_h)
                f = b.font()
                f.setPointSizeF(max(12.0, 12.0 * scale))
                b.setFont(f)
            # Sidebar labels
            for lbl in getattr(self, '_sidebar_labels', []):
                lf = lbl.font()
                lf.setPointSizeF(max(16.0, 16.0 * scale))
                lbl.setFont(lf)
            # Counter label
            if hasattr(self, 'counter_label') and self.counter_label is not None:
                f2 = self.counter_label.font()
                f2.setPointSizeF(max(12.0, 12.0 * scale))
                self.counter_label.setFont(f2)
            # Info panel fonts
            if hasattr(self, 'info_title_right') and self.info_title_right is not None:
                ft = self.info_title_right.font()
                ft.setPointSizeF(max(20.0, 20.0 * scale))
                self.info_title_right.setFont(ft)
            if hasattr(self, 'info_label_right') and self.info_label_right is not None:
                fl = self.info_label_right.font()
                fl.setPointSizeF(max(28.0, 28.0 * scale))
                self.info_label_right.setFont(fl)
        except Exception:
            pass
        super().resizeEvent(event)

    def toggle_theme(self):
        # Toggle between light and dark theme and update relevant widget styles
        self._theme_dark = not getattr(self, '_theme_dark', False)
        if self._theme_dark:
            # Dark theme: dark canvas, dark title and info panel with light text
            try:
                canvas.bgcolor = '#0b0b0b'
            except Exception:
                pass
            self.theme_btn.setText("🌙")
            # Title (canvas_label) dark style
            self.canvas_label.setStyleSheet("""
                QLabel {
                    font-size: 22px;
                    font-weight: bold;
                    padding: 12px;
                    background-color: #111214;
                    color: #f5f5f5;
                    border-bottom: 1px solid #2a2a2a;
                }
            """)
            # Info panel dark style
            self.info_panel_right.setStyleSheet("""
                QFrame { background: #1b1d20; border-left: 1px solid #333333; border-radius:6px; }
                QLabel#InfoTitle { color: #f2f2f2; font-weight: 700; font-size: 16px; }
                QLabel#InfoText { color: #dcdcdc; font-size: 13px; }
            """)
            # Sidebar subtle darken so it matches theme
            self.sidebar.setStyleSheet("""
                QFrame { background: #252628; border-right: 1px solid #3a3a3a; }
                QLabel { font-family: 'Segoe UI', 'Verdana', 'Arial'; color: #f5f5f5; }
            """)
            # Counter label readability on dark
            self.counter_label.setStyleSheet("color: #f5f5f5; font-weight: bold; margin-top: 8px; font-size: 14px; padding:4px;")
        else:
            # Light theme: restore canvas and widget styles to original light appearance
            try:
                canvas.bgcolor = 'white'
            except Exception:
                pass
            self.theme_btn.setText("☼")
            # Restore title style (light)
            self.canvas_label.setStyleSheet("""
                QLabel {
                    font-size: 22px;
                    font-weight: bold;
                    padding: 12px;
                    background-color: #fafafa;
                    color: #222222;
                    border-bottom: 1px solid #e0e0e0;
                }
            """)
            # Restore info panel light style
            self.info_panel_right.setStyleSheet("""
                QFrame { background: #f7f8fa; border-left: 1px solid #dddddd; border-radius:6px; }
                QLabel#InfoTitle { color: #333333; font-weight: 700; font-size: 16px; }
                QLabel#InfoText { color: #444444; font-size: 13px; }
            """)
            # Restore sidebar style
            self.sidebar.setStyleSheet("""
                QFrame { background: #2f3136; border-right: 1px solid #444444; }
                QLabel { font-family: 'Segoe UI', 'Verdana', 'Arial'; color: #f5f5f5; }
            """)
            self.counter_label.setStyleSheet("color: white; font-weight: bold; margin-top: 15px; font-size: 14px; padding:4px;")

    def show_decay_timer(self, seconds, label=None):
        """Show a proper countdown in the decay label and update every second.
        Optional label: 'neutron' | 'proton' | 'electron' or a custom string.
        """
        # Ensure timer object exists and is single shared QTimer
        self._decay_timer_seconds = int(seconds)
        # Choose display text based on label
        if label is None:
            disp = f"Neutron decay in {self._decay_timer_seconds} s"
        else:
            disp_label = str(label).capitalize()
            disp = f"{disp_label} decay in {self._decay_timer_seconds} s"
        self.decay_timer_label.setText(disp)
        self.decay_timer_label.show()
        if hasattr(self, '_decay_timer_qt') and self._decay_timer_qt is not None:
            # stop and reset
            self._decay_timer_qt.stop()
        else:
            self._decay_timer_qt = QTimer(self)
            self._decay_timer_qt.setInterval(1000)
            self._decay_timer_qt.timeout.connect(self._decay_timer_tick)
        self._decay_timer_qt.start()

    def _decay_timer_tick(self):
        # Called every second by the QTimer
        try:
            self._decay_timer_seconds -= 1
        except Exception:
            # Safety: if something went wrong, hide the timer
            self.hide_decay_timer()
            return
        if self._decay_timer_seconds > 0:
            # Keep existing label prefix when updating
            full_text = self.decay_timer_label.text()
            # Try to parse prefix (e.g., "Neutron decay in ") and replace the number
            parts = full_text.rsplit(' ', 2)
            if len(parts) >= 3:
                prefix = ' '.join(parts[:-2]) + ' '
                self.decay_timer_label.setText(f"{prefix}{self._decay_timer_seconds} s")
            else:
                self.decay_timer_label.setText(f"Neutron decay in {self._decay_timer_seconds} s")
        else:
            # Time's up, hide label and stop timer
            self.hide_decay_timer()

    def hide_decay_timer(self):
        # Stop and hide the decay timer cleanly
        if hasattr(self, '_decay_timer_qt') and self._decay_timer_qt is not None:
            self._decay_timer_qt.stop()
        self.decay_timer_label.hide()

    def run_atom_background_task(self):
        # Background task that runs every second if there is an atom
        def check_and_run():
            if len(protons) > 0 and len(neutrons) > 0 and len(electrons) > 0:
                if random.random() < chance_percent / 100.0:
                    explode()
                    print("THE ATOM WAS SPLIT")
                    self.decay_timer_label.setText("Atomic Split")
                    self.decay_timer_label.show()
                    # Hide label after 3 seconds
                    QTimer.singleShot(3000, self.hide_decay_timer)
        self.atom_bg_timer = QTimer(self)
        self.atom_bg_timer.timeout.connect(check_and_run)
        self.atom_bg_timer.start(1000)  # Run every second

if __name__ == "__main__":
    window = AnimatedSidebar()
    canvas.show()
    # Use the Qt application event loop to run the GUI
    sys.exit(qt_app.exec_())