import blenderproc as bproc
import numpy as np
import random
import cv2
import bmesh
import json
import bpy

# to download haven: blenderproc download haven haven
HAVEN_LOC = "/home/tyler/Documents/haven"
RESISTOR_COLORS = {
    "black": [0, 0, 0, 1.0], "brown": [0.15, 0.05, 0.02, 1.0], "red": [0.8, 0, 0, 1.0],
    "orange": [1, 0.3, 0, 1.0], "yellow": [1, 0.8, 0, 1.0], "green": [0, 0.5, 0, 1.0],
    "blue": [0, 0, 0.8, 1.0], "violet": [0.3, 0, 0.5, 1.0], "gray": [0.2, 0.2, 0.2, 1.0],
    "white": [1, 1, 1, 1.0], "gold": [0.8, 0.5, 0.1, 1.0], "silver": [0.6, 0.6, 0.6, 1.0]
}
BODY_COLORS = [[0.82, 0.70, 0.55, 1.0], 
               [0.6, 0.8, 0.9, 1.0], 
               [0.3, 0.5, 0.3, 1.0], 
               [0.7, 0.7, 0.7, 1.0],
               [210/255, 180/255, 140/255, 1.0],
               [60/255, 120/255, 170/255, 1.0],
               [90/255, 45/255, 30/255, 1.0]]


bproc.init()

def add_cylinder(objs, radius, length, mat, z0, vertices=64):
    # default cylinder depth is 2, radius is 1
    obj = bproc.object.create_primitive("CYLINDER", vertices=vertices, end_fill_type='NOTHING')
    obj.set_scale([radius, radius, length / 2.0])         
    obj.set_location([0.0, 0.0, z0 + length / 2.0])  
    obj.replace_materials(mat)
    objs.append(obj)
    return z0 + length

def add_sphere(objs, rx, ry, rz, mat, z0, z_length):
    obj = bproc.object.create_primitive("SPHERE")
    obj.set_scale([rx, ry, rz])
    obj.set_location([0.0, 0.0, z0 + z_length / 2.0])
    obj.replace_materials(mat)
    objs.append(obj)
    return z0 + z_length

def add_frustum(objs, r_start, r_end, length, mat, z0, vertices=64):
    r_start = max(r_start, 1e-9)

    obj = bproc.object.create_primitive(
        "CONE",
        vertices=vertices,
        radius1=1.0,
        radius2=(r_end / r_start),
        depth=2.0,
        end_fill_type='NOTHING'
    )

    obj.set_scale([r_start, r_start, length / 2.0])
    obj.set_location([0.0, 0.0, z0 + length / 2.0])
    obj.replace_materials(mat)
    objs.append(obj)
    return z0 + length

def get_resistor_color(i, ca=None):
    band_color = random.choice(list(RESISTOR_COLORS.keys())) if not ca else ca[i % len(ca)].lower()
    band_value = RESISTOR_COLORS[band_color]
    band_value = [min(1.0, max(0.0, c + random.uniform(-0.05, 0.05))) if i < 3 else c for i,c in enumerate(band_value)]
    band_mat = bproc.material.create(f"band_{i}")
    band_mat.set_principled_shader_value("Base Color", band_value)
    if band_color in ["gold", "silver"] and random.random() < 0.7:
        band_mat.set_principled_shader_value("Metallic", 0.8 + random.uniform(-0.4, 0.2))
        band_mat.set_principled_shader_value("Roughness", 0.6 + random.uniform(-0.2, 0.2))
    else:
        band_mat.set_principled_shader_value("Roughness", 0.3 + random.uniform(-0.2, 0.3))
    return band_mat, band_color

def create_procedural_resistor(order=None):
    # proportions 
    if order:
        band_count = len(order)
    else:
        band_count   = random.randint(3, 6) # there is a 1-band ("zero-ohm"... but it's not a superconductor) resistor.

    band_width   = 0.015 + random.uniform(-0.01, 0.03)         
    base_width   = 0.020 + random.uniform(-0.005, 0.02)         
    lead_length  = 0.8 + random.uniform(-0.1, 0.1)         
    body_radius = 0.065 + random.uniform(-0.01, 0.005)     
    body_inner_radius = 0.055 + random.uniform(-0.005, 0.005)
    lead_radius     = 0.01 + random.uniform(-0.005, 0.005)   

    # materials 
    body_color = random.choice(BODY_COLORS)
    body_mat = bproc.material.create("body_mat")
    body_mat.set_principled_shader_value("Base Color", body_color)
    body_mat.set_principled_shader_value("Roughness", 0.5)

    lead_mat = bproc.material.create("lead_mat")
    lead_color_base_rng = 0.8 + random.uniform(-0.2, 0.1)
    lead_mat.set_principled_shader_value(
        "Base Color",
        [lead_color_base_rng + random.uniform(-0.05,0.05),
         lead_color_base_rng + random.uniform(-0.05,0.05),
         lead_color_base_rng + random.uniform(-0.05,0.05), 1.0]
    )
    lead_mat.set_principled_shader_value("Metallic", 1.0 + random.uniform(-0.1, 0.0))
    lead_mat.set_principled_shader_value("Roughness", 0.4 + random.uniform(-0.35, 0.3))

    # random_design choices and constraining
    has_middle_gap = random.random() < (0.6 + (0.1*(6-band_count))) if band_count < 6 else False
    body_inner_radius = max(min(body_inner_radius, body_radius), body_radius * 0.8)

    # begin building segments
    current_z = 0.0
    objs = []
    band_colors = []

    # lead + rounded cap + body start
    current_z = add_cylinder(objs, lead_radius, lead_length, lead_mat, current_z)
    cap_len = body_radius  
    cap_rz  = cap_len / 2.0
    current_z = add_sphere(objs, body_radius, body_radius, cap_rz, body_mat, current_z, cap_len) - (cap_len/2.0) # remove half, so cap is a hemisphere
    body_len = body_radius * 0.25
    current_z = add_cylinder(objs, body_radius, body_len, body_mat, current_z)

    # first band + taper
    band_mat, band_color = get_resistor_color(0, order)
    current_z = add_cylinder(objs, body_radius, band_width, band_mat, current_z)
    band_colors.append(band_color)
    #taper_len = body_radius * 0.25
    if body_inner_radius < body_radius:
        current_z = add_frustum(objs, body_radius, body_inner_radius, base_width, body_mat, current_z)
    else:
        current_z = add_cylinder(objs, body_radius, base_width, body_mat, current_z)

    # body: band + base + band + ... + band
    mx = band_count-1 if band_count > 3 else band_count
    for i in range(1, mx):
        band_mat, band_color = get_resistor_color(i, order)
        current_z = add_cylinder(objs, body_inner_radius, band_width, band_mat, current_z)
        band_colors.append(band_color)

        if i < mx - 1:
            current_z = add_cylinder(objs, body_inner_radius, base_width, body_mat, current_z)

    if has_middle_gap:
        gap_len = base_width * (6-band_count)
        current_z = add_cylinder(objs, body_inner_radius, gap_len, body_mat, current_z)

    # taper + last band
    if body_inner_radius < body_radius:
        current_z = add_frustum(objs, body_inner_radius, body_radius, base_width, body_mat, current_z)
    else:
        current_z = add_cylinder(objs, body_radius, base_width, body_mat, current_z)
    
    if band_count > 3:
        band_mat, band_color = get_resistor_color(band_count-1, order)
        current_z = add_cylinder(objs, body_radius, band_width, band_mat, current_z)
        band_colors.append(band_color)
    else: # left aligned 3-band
        current_z = add_cylinder(objs, body_radius, band_width, body_mat, current_z) # empty band

    # body end + rounded cap + lead (using beginning code mirrored)
    current_z = add_cylinder(objs, body_radius, body_len, body_mat, current_z)
    current_z = add_sphere(objs, body_radius, body_radius, cap_rz, body_mat, current_z - (cap_len/2.0), cap_len)
    current_z = add_cylinder(objs, lead_radius, lead_length, lead_mat, current_z)
      
    # join into a single object a
    resistor = objs[0]
    resistor.join_with_other_objects(objs[1:])

    # combine faces
    bm = resistor.mesh_as_bmesh(return_copy=True)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-6)
    resistor.update_from_bmesh(bm)

    # add some surface noise / imperfection
    tex = bproc.material.create_procedural_texture("NOISE") 
    resistor.add_displace_modifier( 
        texture=tex, 
        strength=random.uniform(0.001, 0.01), 
        subdiv_level=random.randint(1,3), 
    )
    
    #add some dust
    for mat in resistor.get_materials():
        bproc.material.add_dust(
            mat,
            strength=random.uniform(0,0.3),        
            texture_scale=0.1 + random.uniform(-0.1,0.2)   
        ) 

    # recenter origin on body
    bm = resistor.mesh_as_bmesh(return_copy=True)
    body_z_values = []
    lead_threshold = lead_radius * 1.5

    for v in bm.verts:
        r = (v.co.x * v.co.x + v.co.y * v.co.y) ** 0.5
        if r > lead_threshold:
            body_z_values.append(v.co.z)
    body_center_z = sum(body_z_values) / len(body_z_values)
    for v in bm.verts:
        v.co.z -= body_center_z

    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-6)
    resistor.update_from_bmesh(bm)

    resistor.set_rotation_euler([-np.pi/2, 0, 0])
    resistor.set_shading_mode("SMOOTH")

    return resistor, band_colors

def make_random_resistor(where):
    # clean up scene
    bproc.clean_up(clean_up_camera=True)
    world = bpy.context.scene.world
    world.use_nodes = True
    nodes = world.node_tree.nodes

    # world background
    haven_hdri_path = bproc.loader.get_random_world_background_hdr_img_path_from_haven(HAVEN_LOC)
    bproc.world.set_world_background_hdr_img(haven_hdri_path)

    #resistor, band_colors = create_procedural_resistor(["brown", "black", "red", "gold"])
    resistor, band_colors = create_procedural_resistor()
    base_pos = np.array([0.0, 0.0, 0.0]) 
    pos_noise = np.random.normal(loc=0.0, scale=0.01, size=3)
    pos_noise[2] = 0.0
    resistor.set_location((base_pos + pos_noise).tolist())

    """
    # ground Plane for reference
    base = bproc.object.create_primitive('PLANE')
    base.set_scale([0.5, 0.5, 0.5])   # 0.5 m x 0.5 m, tweak as needed
    base_z = 0.0    # ground level
    base.set_location([0.0, 0.0, base_z])
    base_mat = bproc.material.create("base_mat")
    base_mat.set_principled_shader_value("Base Color", [0.8, 0.8, 0.8, 1.0])
    base_mat.set_principled_shader_value("Roughness", 0.9)
    base.replace_materials(base_mat)
    """

    # random lighting
    light = bproc.types.Light()
    light.set_energy(random.uniform(20, 100))
    light.set_location(bproc.sampler.shell(center=[0, 0, 0], radius_min=2, radius_max=5, elevation_min=20, elevation_max=80))

    # camera randomization
    poi = np.array([0.0, 0.0, 0.0])
    location = bproc.sampler.shell(
        center=poi,
        radius_min=1.5,
        radius_max=3,
        elevation_min=-30,
        elevation_max=60,
        azimuth_min=-40,
        azimuth_max=40
    )
    rotation_matrix = bproc.camera.rotation_from_forward_vec(np.array(poi) - location)
    cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
    bproc.camera.add_camera_pose(cam2world_matrix)


    data = bproc.renderer.render()
    img = data["colors"][0]
    img_bgr = img[:, :, ::-1] # RGB -> BGR for OpenCV
    cv2.imwrite(where, img_bgr)

    resistor.delete()
    light.delete()

    return band_colors

if __name__ == "__main__":
    n=1000
    bands = []
    for i in range(n):
        o = {}
        o["file"] = f"images/{i}.png"
        print(o["file"])
        o["bands"] = make_random_resistor(o["file"])
        bands.append(o)
    with open("bands.json", "w") as jf:
        json.dump(bands, jf, indent=4)
    
