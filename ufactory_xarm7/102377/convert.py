#!/usr/bin/env python3
#python urdf_to_mjcf.py input.urdf output.xml

import sys
import xml.etree.ElementTree as ET

def parse_origin(origin_element):
    xyz = [0.0, 0.0, 0.0]
    rpy = [0.0, 0.0, 0.0]

    if origin_element is not None:
        if 'xyz' in origin_element.attrib:
            xyz_str = origin_element.attrib['xyz'].split()
            xyz = [float(v) for v in xyz_str]
        if 'rpy' in origin_element.attrib:
            rpy_str = origin_element.attrib['rpy'].split()
            rpy = [float(v) for v in rpy_str]
    return {'xyz': xyz, 'rpy': rpy}


def parse_geometry(geometry_element):
    mesh_filename = None
    mesh_elem = geometry_element.find('mesh')
    if mesh_elem is not None and 'filename' in mesh_elem.attrib:
        mesh_filename = mesh_elem.attrib['filename']
    return mesh_filename


def parse_link(link_element):
    link_name = link_element.attrib['name']

    visuals = []
    for visual_elem in link_element.findall('visual'):
        origin_elem = visual_elem.find('origin')
        origin = parse_origin(origin_elem)
        geometry_elem = visual_elem.find('geometry')
        mesh_filename = parse_geometry(geometry_elem)
        if mesh_filename:
            visuals.append({
                'mesh': mesh_filename,
                'origin': origin
            })

    collisions = []
    for coll_elem in link_element.findall('collision'):
        origin_elem = coll_elem.find('origin')
        origin = parse_origin(origin_elem)
        geometry_elem = coll_elem.find('geometry')
        mesh_filename = parse_geometry(geometry_elem)
        if mesh_filename:
            collisions.append({
                'mesh': mesh_filename,
                'origin': origin
            })

    return {
        'name': link_name,
        'visuals': visuals,
        'collisions': collisions
    }


def parse_joint(joint_element):
    joint_name = joint_element.attrib['name']
    joint_type = joint_element.attrib['type']  # 'revolute', 'prismatic', or 'fixed'
    
    origin_elem = joint_element.find('origin')
    origin = parse_origin(origin_elem)

    axis_elem = joint_element.find('axis')
    axis = [1.0, 0.0, 0.0]
    if axis_elem is not None and 'xyz' in axis_elem.attrib:
        axis = [float(v) for v in axis_elem.attrib['xyz'].split()]

    parent_elem = joint_element.find('parent')
    parent_link = parent_elem.attrib['link'] if parent_elem is not None else None

    child_elem = joint_element.find('child')
    child_link = child_elem.attrib['link'] if child_elem is not None else None

    limit_elem = joint_element.find('limit')
    lower = None
    upper = None
    if limit_elem is not None:
        if 'lower' in limit_elem.attrib:
            lower = float(limit_elem.attrib['lower'])
        if 'upper' in limit_elem.attrib:
            upper = float(limit_elem.attrib['upper'])

    return {
        'name': joint_name,
        'type': joint_type,
        'origin': origin,     # {xyz, rpy}
        'axis': axis,
        'parent': parent_link,
        'child': child_link,
        'limit': (lower, upper)
    }


def load_urdf(filename):
    tree = ET.parse(filename)
    root = tree.getroot()

    robot_name = root.attrib.get('name', 'robot')
    
    links_dict = {}
    for link_elem in root.findall('link'):
        link_info = parse_link(link_elem)
        links_dict[link_info['name']] = link_info

    joints_dict = {}
    for joint_elem in root.findall('joint'):
        joint_info = parse_joint(joint_elem)
        joints_dict[joint_info['name']] = joint_info

    return {
        'robot_name': robot_name,
        'links': links_dict,
        'joints': joints_dict
    }


def build_mjcf_string(urdf_data):
    robot_name = urdf_data['robot_name']
    links = urdf_data['links']
    joints = urdf_data['joints']

    mesh_set = set()
    for l in links.values():
        for v in l['visuals']:
            mesh_set.add(v['mesh'])
        for c in l['collisions']:
            mesh_set.add(c['mesh'])
    mesh_list = sorted(list(mesh_set))

    lines = []
    lines.append('<?xml version="1.0"?>')
    lines.append(f'<mujoco model="{robot_name}">')
    lines.append('    <!-- Compiler Options -->')
    lines.append('    <compiler angle="radian" coordinate="local" />')
    lines.append('')
    lines.append('    <!-- Simulation Options -->')
    lines.append('    <option gravity="0 0 -9.81" />')
    lines.append('')
    lines.append('    <!-- Asset Definitions -->')
    lines.append('    <asset>')
    lines.append('        <!-- Mesh Assets -->')
    for mesh_file in mesh_list:
        short_name = mesh_file.replace('.obj','').replace('-','_').replace('/','_')
        lines.append(f'        <mesh name="{short_name}" file="{mesh_file}" />')
    lines.append('    </asset>')
    lines.append('')

    lines.append('    <!-- World Body -->')
    lines.append('    <worldbody>')
    lines.append('        <!-- Base Body -->')
    lines.append('        <body name="base">')
    children_map = {}
    for joint_name, joint_info in joints.items():
        p = joint_info['parent']
        if p not in children_map:
            children_map[p] = []
        children_map[p].append(joint_name)

    def print_body(link_name, indent="            "):
        if link_name not in children_map:
            return

        for jname in children_map[link_name]:
            jinfo = joints[jname]
            child_link_name = jinfo['child']
            child_link_data = links[child_link_name]
            joint_type = jinfo['type']
            origin_xyz = jinfo['origin']['xyz']
            origin_rpy = jinfo['origin']['rpy']
            axis = jinfo['axis']
            lower, upper = jinfo['limit']

            if joint_type == 'revolute':
                mj_type = 'hinge'
            elif joint_type == 'prismatic':
                mj_type = 'slide'
            else:  # 'fixed'
                mj_type = 'free'

            pos_str = f"{origin_xyz[0]} {origin_xyz[1]} {origin_xyz[2]}"

            euler_str = f"{origin_rpy[0]} {origin_rpy[1]} {origin_rpy[2]}"

            lines.append(f'{indent}<!-- Joint {jname}: {joint_type} connecting {link_name} -> {child_link_name} -->')
            lines.append(f'{indent}<body name="{child_link_name}" pos="{pos_str}" euler="{euler_str}">')

            if joint_type != 'fixed':
                range_str = ""
                if lower is not None and upper is not None:
                    range_str = f' range="{lower} {upper}"'
                axis_str = f"{axis[0]} {axis[1]} {axis[2]}"
                lines.append(f'{indent}    <joint name="{jname}" type="{mj_type}" axis="{axis_str}"{range_str} />')

            for vis in child_link_data['visuals']:
                vx, vy, vz = vis['origin']['xyz']
                geom_pos = f"{vx} {vy} {vz}"
                short_name = vis['mesh'].replace('.obj','').replace('-','_').replace('/','_')
                lines.append(f'{indent}    <geom type="mesh" mesh="{short_name}" pos="{geom_pos}" />')

            print_body(child_link_name, indent + "    ")

            lines.append(f'{indent}</body>')

    base_link_data = links.get('base', None)
    print_body("base", "            ")

    lines.append('        </body>')
    lines.append('    </worldbody>')
    lines.append('</mujoco>')

    return "\n".join(lines)


def main():
    if len(sys.argv) < 3:
        print("Usage: python urdf_to_mjcf.py input.urdf output.xml")
        sys.exit(1)

    input_urdf = sys.argv[1]
    output_mjcf = sys.argv[2]
    urdf_data = load_urdf(input_urdf)
    mjcf_str = build_mjcf_string(urdf_data)

    with open(output_mjcf, 'w') as f:
        f.write(mjcf_str)
    print(f"Converted {input_urdf} to {output_mjcf}.")

if __name__ == "__main__":
    main()
