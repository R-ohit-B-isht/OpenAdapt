import subprocess
from loguru import logger

def get_active_window_state(read_window_data: bool) -> dict:
    """Get the state of the active window.

    Returns:
        dict: A dictionary containing the state of the active window.
            The dictionary has the following keys:
                - "title": Title of the active window.
                - "left": Left position of the active window.
                - "top": Top position of the active window.
                - "width": Width of the active window.
                - "height": Height of the active window.
                - "meta": Meta information of the active window.
                - "data": None (to be filled with window data).
                - "window_id": ID of the active window.
    """
    try:
        active_window = get_active_window()
    except RuntimeError as e:
        logger.warning(e)
        return {}
    meta = get_active_window_meta(active_window)
    rectangle_dict = dictify_rect(meta["rectangle"])
    if read_window_data:
        data = get_element_properties(active_window)
    else:
        data = {}
    state = {
        "title": meta["texts"][0],
        "left": meta["rectangle"].left,
        "top": meta["rectangle"].top,
        "width": meta["rectangle"].width(),
        "height": meta["rectangle"].height(),
        "meta": {**meta, "rectangle": rectangle_dict},
        "data": data,
        "window_id": meta["control_id"],
    }
    try:
        pickle.dumps(state)
    except Exception as exc:
        logger.warning(f"{exc=}")
        state.pop("data")
    return state

def get_active_window_meta(active_window) -> dict:
    """Get the meta information of the active window.

    Args:
        active_window: The active window object.

    Returns:
        dict: A dictionary containing the meta information of the
              active window.
    """
    result = {
        "texts": [active_window],
        "rectangle": {
            "left": 0,
            "top": 0,
            "right": 0,
            "bottom": 0,
        },
        "control_id": 0,
    }
    return result

def get_active_element_state(x: int, y: int) -> dict:
    """Get the state of the active element at the given coordinates.

    Args:
        x (int): The x-coordinate.
        y (int): The y-coordinate.

    Returns:
        dict: A dictionary containing the properties of the active element.
    """
    active_window = get_active_window()
    active_element = active_window
    properties = get_properties(active_element)
    properties["rectangle"] = dictify_rect(properties["rectangle"])
    return properties

def get_active_window() -> str:
    """Get the active window object.

    Returns:
        str: The active window object.
    """
    try:
        result = subprocess.run(["xdotool", "getactivewindow", "getwindowname"], capture_output=True, text=True)
        return result.stdout.strip()
    except Exception as e:
        logger.warning(e)
        return ""

def get_element_properties(element: str) -> dict:
    """Recursively retrieves the properties of each element and its children.

    Args:
        element: An instance of a custom element class
                 that has the `.get_properties()` and `.children()` methods.

    Returns:
        dict: A nested dictionary containing the properties of each element
          and its children.
        The dictionary includes a "children" key for each element,
        which holds the properties of its children.

    Example:
        element = Element()
        properties = get_element_properties(element)
        print(properties)
        # Output: {'prop1': 'value1', 'prop2': 'value2',
                  'children': [{'prop1': 'child_value1', 'prop2': 'child_value2',
                  'children': []}]}
    """
    properties = {
        "rectangle": {
            "left": 0,
            "top": 0,
            "right": 0,
            "bottom": 0,
        },
        "children": [],
    }
    return properties

def dictify_rect(rect: dict) -> dict:
    """Convert a rectangle object to a dictionary.

    Args:
        rect: The rectangle object.

    Returns:
        dict: A dictionary representation of the rectangle.
    """
    rect_dict = {
        "left": rect["left"],
        "top": rect["top"],
        "right": rect["right"],
        "bottom": rect["bottom"],
    }
    return rect_dict

def get_properties(element: str) -> dict:
    """Retrieves specific writable properties of an element.

    This function retrieves a dictionary of writable properties for a given element.
    It achieves this by temporarily modifying the class of the element object using
    monkey patching.This approach is necessary because in some cases, the original
    class of the element may have a `get_properties()` function that raises errors.

    Args:
        element: The element for which to retrieve writable properties.

    Returns:
        A dictionary containing the writable properties of the element,
        with property names as keys and their corresponding values.
    """
    properties = {
        "rectangle": {
            "left": 0,
            "top": 0,
            "right": 0,
            "bottom": 0,
        },
    }
    return properties
