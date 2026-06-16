blender_single_edit_script = """
import bpy
import sys
from pathlib import Path

argv = sys.argv
if '--' in argv:
    argv = argv[argv.index('--') + 1:]
else:
    argv = []

if len(argv) < 2:
    raise RuntimeError('Expected two arguments: input_path output_path')

input_path = Path(argv[0])
output_path = Path(argv[1])

class SURFILE_PT_point_cloud_export(bpy.types.Panel):
    bl_label = 'Surfile Point Cloud'
    bl_idname = 'SURFILE_PT_point_cloud_export'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Surfile'

    def draw(self, context):
        layout = self.layout
        layout.label(text='Edit the point cloud, then export:')
        layout.operator('surfile.export_point_cloud', text='Export Point Cloud and Quit')

class SURFILE_OT_export_point_cloud(bpy.types.Operator):
    bl_idname = 'surfile.export_point_cloud'
    bl_label = 'Export Point Cloud and Quit'

    def execute(self, context):
        obj = context.active_object
        if obj is None:
            self.report({'ERROR'}, 'Please select the imported object before exporting.')
            return {'CANCELLED'}

        bpy.ops.object.mode_set(mode='OBJECT')
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        context.view_layer.objects.active = obj

        try:
            # FIX: Updated parameter signatures to match Blender 4.0/5.x native C++ engine
            bpy.ops.wm.ply_export(
                filepath=str(output_path),
                ascii_format=True,
                export_selected_objects=True,
                apply_modifiers=True,
            )
        except Exception as exc:
            self.report({'ERROR'}, f'Export failed: {exc}')
            return {'CANCELLED'}

        bpy.ops.wm.quit_blender()
        return {'FINISHED'}

def delayed_initialization():
    # 1. Safely delete any default template meshes/objects without breaking context window properties
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for mesh in list(bpy.data.meshes):
        bpy.data.meshes.remove(mesh)

    # 2. Run the import now that the window is fully up, active, and drawn
    try:
        bpy.ops.wm.ply_import(filepath=str(input_path))
    except Exception as exc:
        print(f"Deferred initialization import failed: {exc}")
        
    return None  # Timers returning None run exactly once and stop

# Register UI items normally at top level
bpy.utils.register_class(SURFILE_PT_point_cloud_export)
bpy.utils.register_class(SURFILE_OT_export_point_cloud)

# Defer the clean up and import call until after the UI event loop registers a live window frame
bpy.app.timers.register(delayed_initialization, first_interval=0.1)
"""