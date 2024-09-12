import platform
import cv2
import mido


def intro():
    # SETUP
    print("Wecome to CV-MIDI.")
    print()
    print("MIDI PORT SETUP")
    print()
    # platform detection
    print("Detecting OS/platform...")
    system = platform.system()
    answer = input(
        f"""Your current platform seems to be {system}. 
    This process sets up a MIDI port based on your platform. 
    Would you like to manually configure the MIDI port, regardless of platform?  [y/N (default)]"""
    )
    print()
    midi_port = 0

    # if the user wants to do manual configuration, or if its required on windows
    if (answer.lower() in ["y", "yes"]) or (system == "Windows"):
        if system == "Windows":
            print(
                """WARNING: Virtual MIDI ports are not supported by Windows natively:
            1. You will have to choose configure your MIDI port manually.
            2. Please install a tool such as loopMIDI (https://www.tobias-erichsen.de/software/loopmidi.html) first."""
            )
        midi_port = manual_port_setup()
    # the empty string makes this the default
    elif (answer.lower() in ["n", "no", ""]) and (system != "Windows"):
        midi_port = open_virtual_port()

    print("A MIDI port has been successfully configured.")
    print()
    print("Next, a list of detected video inputs will appear.")
    # simply to pause execution and not bombard the user with info
    _ = input("Press any key to continue:")

    available, _ = get_camera_status()
    num = 0
    looping = True
    # simple input validation
    while looping:
        num = int(input("Enter the video capture port you'd like to use:\n"))
        target_index = num - 1
        if target_index < 0 or target_index >= len(available):
            print("Invalid input. Try again.")
        else:
            looping = False

    # return a tuple of the midi port and the webcam index to open
    return midi_port, num


def manual_port_setup():
    print("Detecting current ports...")
    current_midi_inputs = mido.get_input_names()

    for index, value in enumerate(current_midi_inputs, start=1):
        print(f"{index}. {value}")

    num = 0
    looping = True
    # simple input validation
    while looping:
        num = int(input("Enter the port number you'd like to use:\n"))
        target_index = num - 1
        if target_index < 0 or target_index >= len(current_midi_inputs):
            print("Invalid input. Try again.")
        else:
            looping = False

    midi_port = mido.open_output(current_midi_inputs[num - 1])
    return midi_port


def open_virtual_port():
    print("""Creating a new Virtual MIDI Port...
    If this fails for any reason, try manual MIDI setup.""")

    midi_port = mido.open_output("CV-MIDI (Virtual)", virtual=True)
    return midi_port


# modified version of an answer from SO:
# https://stackoverflow.com/questions/57577445/list-available-cameras-opencv-python
def get_camera_status():
    """
    Test all camera ports and returns a tuple with the available ports
    and the ones that are working.
    """
    is_working = True
    dev_port = 0
    working_ports = []
    available_ports = []
    while is_working:
        camera = cv2.VideoCapture(dev_port)
        if not camera.isOpened():
            is_working = False
            print(f"{dev_port}. [❌] Port is NOT WORKING.")
        else:
            # we don't care about the image, just if it's reading or not
            is_reading, _ = camera.read()
            w = camera.get(3)
            h = camera.get(4)
            if is_reading:
                print(f"{dev_port}. [✅] Port is working and reads images ({h} x {w})")
                working_ports.append(dev_port)
            else:
                print(f"{dev_port}. [❌] Port ({h} x {w} is present but DOES NOT READ)")
                available_ports.append(dev_port)
        dev_port += 1
    return available_ports, working_ports
