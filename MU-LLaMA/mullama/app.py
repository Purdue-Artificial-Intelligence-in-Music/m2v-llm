import argparse
import threading
import time
from dataclasses import dataclass

import dearpygui.dearpygui as dpg
import numpy as np
import PIL.Image as Image
from mullm import DefaultGenConfig, ImageGenerator

IMG_HEIGHT = 512
IMG_WIDTH = 512
MENU_HEIGHT = 96
MENU_WIDTH = 512

VP_WIDTH = IMG_WIDTH
VP_HEIGHT = IMG_HEIGHT + MENU_HEIGHT

# GENERATE_BUTTON_WIDTH = 128
# PROGRESS_BAR_WIDTH = MENU_WIDTH - GENERATE_BUTTON_WIDTH - dpg.mvStyleVar_ItemSpacing


@dataclass
class State:
    running: bool = False
    generator: "str" = "random"
    audio_path: "str" = None


# This is just used to update the texture.
def update_image(img):
    # update the texture by setting a new value of the dynamic texture.
    # dynamic texture is in RGBA format.
    dpg.set_value("texture", img)


def generate_image_thread(sender):
    print("Generating image with prompt:", DEFAULT_CONFIG.prompt)
    if STATE.generator == "random":
        img = np.random.random((IMG_HEIGHT * IMG_WIDTH * 4))  # 4 since RGBA.
        time.sleep(0.5)
    elif STATE.generator == "mullm":
        # TODO: Do proper exception handling.
        try: 
            img = IMAGE_GENERATOR(STATE.audio_path)
            img = img.convert("RGBA")
            img = np.array(img).flatten() / 255.0
        except:
            print("Error generating image.")
            STATE.running = False
            dpg.enable_item(sender)
            dpg.set_item_label(sender, "generate")
            return 
        print("Image generated.")
    
    update_image(img)
    STATE.running = False

    # re-enable the generate button and revert its label.
    dpg.enable_item(sender)
    dpg.set_item_label(sender, "generate")

    print("Image generated.")


def prompt_update(sender, app_data):
    DEFAULT_CONFIG.prompt = app_data


def generate(sender, app_data):
    if STATE.running:
        return
    else:
        # disable the generate button and change its label.
        dpg.disable_item(sender)
        dpg.set_item_label(sender, "generating...")
        # update state and start thread
        STATE.running = True

        # if-else to check if we are using mullm an audio file is selected.
        if STATE.generator == "mullm" and STATE.audio_path is None:
            print("No audio file selected.")
            STATE.running = False
            dpg.enable_item(sender)
            dpg.set_item_label(sender, "generate")
            return
        else:
            thread = threading.Thread(
                target=generate_image_thread, args=(sender,), daemon=True
            )
            thread.start()


def open_file_dialog(sender, app_data):
    dpg.show_item("file_dialog_id")


def file_callback(sender, app_data):
    audio_paths = list(app_data["selections"].values())
    STATE.audio_path = audio_paths[0]
    print("File Selected: ", STATE.audio_path)


def file_cancel_callback(sender, app_data):
    print("Cancel was clicked.")
    print("Sender: ", sender)
    print("App Data: ", app_data)


def create_app():
    with dpg.window(
        tag="app window",
        width=VP_WIDTH,
        height=VP_HEIGHT,
        no_move=True,
        no_resize=True,
        no_title_bar=True,
        pos=(0, 0),
    ):
        # Image window
        with dpg.child_window(tag="img", width=IMG_WIDTH, height=IMG_HEIGHT):
            # Create random texture as default.
            # It is dynamic so we can update it in the callback.
            with dpg.texture_registry(tag="texreg"):
                # add dynamic texture, this is in RGBA format.
                dpg.add_dynamic_texture(
                    width=IMG_HEIGHT,
                    height=IMG_WIDTH,
                    default_value=np.random.random((IMG_HEIGHT * IMG_WIDTH * 4)),
                    tag="texture",
                )
            # add image with the dynamic texture.
            dpg.add_image(
                texture_tag="texture", height=IMG_HEIGHT, width=IMG_WIDTH, tag="image"
            )

        # Menu window
        with dpg.child_window(tag="menu", width=MENU_WIDTH, height=MENU_HEIGHT):

            # This is file dialog stuff.
            with dpg.file_dialog(
                directory_selector=False,
                show=False,
                callback=file_callback,
                id="file_dialog_id",
                width=IMG_WIDTH,
                height=IMG_HEIGHT,
            ):
                dpg.add_file_extension(
                    ".wav",
                    color=(0, 255, 128, 255),
                )
                dpg.add_file_extension(
                    ".mp3",
                    color=(0, 255, 128, 255),
                )
            dpg.add_button(
                label="Select Audio File",
                callback=open_file_dialog,
                width=-1,
                height=24,
            )

            # Prompt stuff
            dpg.add_input_text(
                hint="Prompt",
                width=-1,
                height=24,
                callback=prompt_update,
                default_value=DEFAULT_CONFIG.prompt,
            )

            # Generate button
            with dpg.group(horizontal=True):
                #     progress_bar = dpg.add_progress_bar(
                #         default_value=0, overlay="0%", height=32, width=PROGRESS_BAR_WIDTH
                #     )
                dpg.add_button(
                    label="generate",
                    callback=generate,
                    height=44,
                    width=-1,
                )

        # Theme stuff
        with dpg.theme() as global_theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_style(
                    dpg.mvStyleVar_WindowPadding, 0, category=dpg.mvThemeCat_Core
                )
        dpg.bind_theme(global_theme)


def main(args):
    # must create a context.
    dpg.create_context()

    # create global state.
    global STATE
    STATE = State()
    STATE.running = False
    STATE.generator = args.generator

    # create global default config.
    global DEFAULT_CONFIG
    DEFAULT_CONFIG = DefaultGenConfig()

    if STATE.generator == "mullm":
        global IMAGE_GENERATOR
        IMAGE_GENERATOR = ImageGenerator(default_config=DEFAULT_CONFIG)

    # set global font scale.
    dpg.set_global_font_scale(1.25)

    # show item registry if needed.
    if args.item_registry:
        dpg.show_item_registry()
    if args.style_editor:
        dpg.show_style_editor()
    if args.debug:
        dpg.show_debug()

    # we may need to resize the viewport when using any of the above.
    resizeable = args.resizable or args.style_editor or args.item_registry or args.debug

    # Create app
    create_app()

    # must create and show viewport.
    dpg.create_viewport(
        title="Image Generator",
        width=VP_WIDTH + 8,
        # max_width=VP_WIDTH + 1,
        height=VP_HEIGHT + 8,
        # max_height=VP_HEIGHT + 1,
        resizable=resizeable,
    )
    dpg.show_viewport()
    dpg.set_primary_window("app window", True)

    # must setup and start dpg
    dpg.setup_dearpygui()
    dpg.start_dearpygui()

    # must destroy the context.
    dpg.destroy_context()


def parse_args():
    parser = argparse.ArgumentParser()

    # These arguments are debug options
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--item-registry", action="store_true", default=False)
    parser.add_argument("--style-editor", action="store_true", default=False)
    parser.add_argument("--resizable", action="store_true", default=False)

    parser.add_argument("--generator", choices=["random", "mullm"], default="random")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)