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
from math import cos, sin, pi
from vispy.color import Color
import random

qt_app = QtWidgets.QApplication(sys.argv)

canvas = scene.SceneCanvas(keys='interactive', size=(1000, 600), show=False)
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
            np.array(trail_points), color=(0.6, 0.6, 1, 0.4),
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
    # If neutron-proton difference > 10, show a 10-second timer, then animate decay
    if abs(n - p) > 10:
        if 'window' in globals() and window:
            window.show_decay_timer(10)
        def start_decay():
            if 'window' in globals() and window:
                window.hide_decay_timer()
            animate_neutron_decay(target_diff=3)
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

    # Basic element names
    names = {
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

    base_name = names.get(p, f"Element Z={p}")

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
        self.sidebar_width = 200
        self.init_ui()
        self.counter_label = QLabel()
        self.counter_label.setStyleSheet("color: white; font-weight: bold; margin-top: 15px;")
        self.counter_label.setWordWrap(True)
        self.sidebar_layout.addWidget(self.counter_label)
        self.update_counter()
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

        self.close_btn = QPushButton("✕")
        self.close_btn.setFixedSize(25, 25)
        self.close_btn.clicked.connect(QApplication.quit)

        self.toggle_btn = QPushButton("☰")
        self.toggle_btn.setFixedSize(25, 25)
        self.toggle_btn.clicked.connect(self.toggle_sidebar)

        toggle_layout = QVBoxLayout()
        toggle_layout.addWidget(self.close_btn)
        toggle_layout.addWidget(self.toggle_btn)
        toggle_layout.addStretch()

        toggle_widget = QWidget()
        toggle_widget.setLayout(toggle_layout)
        toggle_widget.setFixedWidth(40)
        toggle_widget.setStyleSheet("background-color:#16171a;")

        self.sidebar = QFrame()
        self.sidebar.setMaximumWidth(self.sidebar_width)
        self.sidebar.setMinimumWidth(0)
        self.sidebar.setStyleSheet("background-color: #404145; border-right: 1px solid #ccc;")
        self.sidebar_layout = QVBoxLayout(self.sidebar)
        self.sidebar_layout.setContentsMargins(10, 10, 10, 10)
        self.sidebar_layout.setSpacing(10)

        def create_button_row(label_text, add_callback, remove_callback):
            label = QLabel(label_text)
            label.setStyleSheet("color: white; font-weight: bold;")
            self.sidebar_layout.addWidget(label)

            add_container = QWidget()
            add_layout = QHBoxLayout(add_container)
            add_layout.setContentsMargins(0, 0, 0, 0)
            for text, cb in [("+1", add_callback), ("+10", add_callback), ("+50", add_callback)]:
                btn = QPushButton(text)
                btn.setStyleSheet("""
                    QPushButton {
                        background-color: #8FBEED;
                        color: white;
                        border-radius: 3px;
                        padding: 5px 10px;
                    }
                    QPushButton:hover {
                        background-color: #45a049;
                    }
                """)
                amount = int(text.replace("+", ""))
                btn.clicked.connect(lambda checked, a=amount, cb=cb: cb(a))
                add_layout.addWidget(btn)
            self.sidebar_layout.addWidget(add_container)

            remove_container = QWidget()
            remove_layout = QHBoxLayout(remove_container)
            remove_layout.setContentsMargins(0, 0, 0, 0)
            for text, cb in [("-1", remove_callback), ("-10", remove_callback), ("-50", remove_callback)]:
                btn = QPushButton(text)
                btn.setStyleSheet("""
                    QPushButton {
                        background-color: #e57373;
                        color: white;
                        border-radius: 3px;
                        padding: 5px 10px;
                    }
                    QPushButton:hover {
                        background-color: #b71c1c;
                    }
                """)
                amount = int(text.replace("-", ""))
                btn.clicked.connect(lambda checked, a=amount, cb=cb: cb(a))
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
        self.canvas_label.setStyleSheet("""
            QLabel {
                font-size: 18px;
                font-weight: bold;
                padding: 10px;
                background-color: #f0f0f0;
                border-bottom: 1px solid #ccc;
            }
        """)
        content_layout.addWidget(self.canvas_label)
        # Create decay timer label (but don't show yet)
        self.decay_timer_label = QLabel()
        self.decay_timer_label.setAlignment(Qt.AlignCenter)
        self.decay_timer_label.setStyleSheet("font-size: 20px; color: #d32f2f; font-weight: bold; background: #fffbe6; border: 2px solid #d32f2f; border-radius: 8px; padding: 8px; margin: 10px;")
        self.decay_timer_label.setMinimumWidth(180)
        self.decay_timer_label.hide()
        content_layout.addWidget(self.decay_timer_label)
        # ----------------------

        content_layout.addWidget(canvas.native)

        self.main_layout.addWidget(toggle_widget)
        self.main_layout.addWidget(self.sidebar)
        self.main_layout.addWidget(self.content)

        self.animation = QPropertyAnimation(self.sidebar, b"maximumWidth")
        self.animation.setDuration(300)
        self.animation.setEasingCurve(QEasingCurve.OutCubic)

    def toggle_sidebar(self):
        if self.sidebar_expanded:
            self.animation.setStartValue(self.sidebar_width)
            self.animation.setEndValue(0)
        else:
            self.animation.setStartValue(0)
            self.animation.setEndValue(self.sidebar_width)
        self.animation.start()
        self.sidebar_expanded = not self.sidebar_expanded

    def show_decay_timer(self, seconds):
        self.decay_timer_label.show()
        self._decay_timer_seconds = seconds
        self.decay_timer_label.setText(f"Neutron decay in {seconds} s")
        # Cancel any previous timer
        if hasattr(self, '_decay_timer_qt') and self._decay_timer_qt is not None:
            self._decay_timer_qt.stop()
        def update_label():
            self._decay_timer_seconds -= 1
            if self._decay_timer_seconds > 0:
                self.decay_timer_label.setText(f"Neutron decay in {self._decay_timer_seconds} s")
                self._decay_timer_qt = QTimer()
                self._decay_timer_qt.setSingleShot(True)
                self._decay_timer_qt.timeout.connect(update_label)
                self._decay_timer_qt.start(1000)
            else:
                self.decay_timer_label.hide()  # Hide label when countdown finishes
        self._decay_timer_qt = QTimer()
        self._decay_timer_qt.setSingleShot(True)
        self._decay_timer_qt.timeout.connect(update_label)
        self._decay_timer_qt.start(1000)

    def hide_decay_timer(self):
        self.decay_timer_label.hide()
        if hasattr(self, '_decay_timer_qt') and self._decay_timer_qt is not None:
            self._decay_timer_qt.stop()
            self._decay_timer_qt = None

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
    app.run()
