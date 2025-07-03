import os
import re
import json
import networkx as nx
from pathlib import Path
import numpy as np
from matplotlib import cm
import cmcrameri.cm as cmc
from matplotlib.colors import to_hex


def find_block_end(content, block_start, block_end):
    # Find body based on indentation
    block_body_lines = content[block_start:block_end].splitlines()
    block_indent_level = None
    for i, line in enumerate(block_body_lines):
        # Skip empty lines
        stripped_line = line.strip()
        if stripped_line == "" or stripped_line.startswith("#"):
            continue

        # Determine the indent level of the first non-empty line
        if block_indent_level is None:
            block_indent_level = len(line) - len(line.lstrip())

        # Check if the current line has less indentation, indicating the end of the class body
        current_indent_level = len(line) - len(line.lstrip())
        if current_indent_level < block_indent_level:
            block_end = content.index(block_body_lines[i])  # Update the position of class_end
            return block_end

    return block_end


def generate_canvas_for_python_file_coloured(python_file, canvas_file, max_func_call=8, dir='TheVault',
                                             canvas_width=3000, canvas_height=1080,
                                             levels_iters=5, update_layout=False):
    """
    Generate a canvas JSON for visualizing function and method calls in a Python file with a hierarchical layout
    adjusted to maintain a 16:9 aspect ratio. Reuses node positions from an existing canvas file if available.

    Args:
        python_file (str): The path to the Python file to process.
        canvas_file (str): The name of the canvas JSON file to save.
        max_func_call (int): The maximum number of times a function can be called before being excluded.
        dir (str): The directory under which to save the generated canvas.
        canvas_width (int): The approximate width of the canvas (for aspect ratio).
        canvas_height (int): The approximate height of the canvas (for aspect ratio).
    """
    # Ensure the directory exists
    Path(dir).mkdir(parents=True, exist_ok=True)
    canvas_file_path = os.path.join(dir, canvas_file)

    # Check if the canvas file already exists and load node positions if available
    existing_positions = {}
    if os.path.exists(canvas_file_path) and (not update_layout):
        with open(canvas_file_path, 'r') as existing_file:
            existing_canvas = json.load(existing_file)
            existing_positions = {
                node['text']: (node['x'], node['y'])
                for node in existing_canvas.get('nodes', [])
                if 'text' in node
            }

    # Read the contents of the Python file
    with open(python_file, 'r') as file:
        content = file.read()

    # Patterns to parse functions, classes, and methods
    function_pattern = re.compile(r'^def\s+(\w+)\(([^)]*)\):', re.MULTILINE)
    #     class_pattern = re.compile(r'^class\s+(\w+)\s*:', re.MULTILINE)
    class_pattern = re.compile(r'^class\s+(\w+)(\(([^)]*)\))?\s*:', re.MULTILINE)
    method_pattern = re.compile(r'^\s+def\s+(\w+)\(([^)]*)\):', re.MULTILINE)
    call_pattern = re.compile(r'\b(\w+)\s*\(')

    # Dictionary to store functions/methods and their call relationships
    functions = {}
    class_methods = {}

    # Parse global functions
    for match in function_pattern.finditer(content):
        function_name = match.group(1)
        args = match.group(2).replace(' ', '').split(',') if match.group(2) else []
        start = match.end()
        next_match = function_pattern.search(content, start)
        end = next_match.start() if next_match else len(content)

        end = find_block_end(content=content, block_start=start, block_end=end)

        body = content[start:end]

        # Find called functions
        called_functions = set(call_match.group(1) for call_match in call_pattern.finditer(body))
        functions[function_name] = {
            'args': args,
            'calls': called_functions
        }

    # Parse classes and their methods
    for class_match in class_pattern.finditer(content):
        class_name = class_match.group(1)
        class_start = class_match.end()
        next_class = class_pattern.search(content, class_start)
        class_end = next_class.start() if next_class else len(content)

        class_end = find_block_end(content=content, block_start=class_start, block_end=class_end)

        # Get the class body slice
        class_body = content[class_start:class_end]

        # Handle methods in the class
        methods = {}
        for method_match in method_pattern.finditer(class_body):
            method_name = method_match.group(1)
            full_method_name = f"{class_name}.{method_name}"
            args = method_match.group(2).replace(' ', '').split(',') if method_match.group(2) else []
            method_start = method_match.end()
            next_method = method_pattern.search(class_body, method_start)
            method_end = next_method.start() if next_method else len(class_body)

            method_end = find_block_end(content=content, block_start=method_start, block_end=method_end)

            method_body = class_body[method_start:method_end]

            # Find called functions
            called_functions = set(call_match.group(1) for call_match in call_pattern.finditer(method_body))
            methods[full_method_name] = {
                'args': args,
                'calls': called_functions
            }
        class_methods[class_name] = methods
        functions.update(methods)

        # Create a mapping of short method names to full names
        method_mapping = {}

        # Populate method_mapping for global functions
        for func_name in list(functions.keys()):  # Use list to avoid modifying the dictionary during iteration
            if "." not in func_name:  # Skip class methods
                method_mapping[func_name] = func_name

        # Populate method_mapping for class methods
        for class_name, methods in class_methods.items():
            for method_name in list(methods.keys()):  # Iterate over class methods
                short_method_name = method_name.split(".")[-1]  # Extract short method name
                if short_method_name in functions:  # Check if the short method name exists in functions
                    # Map the short name to the full class-specific name
                    method_mapping[short_method_name] = method_name
                    # Remove the short name from the functions dictionary to avoid duplication
                    del functions[short_method_name]
                else:
                    method_mapping[short_method_name] = method_name

        print(f'{method_mapping=}')

        # Update `calls` with full method names
        for function_name, details in functions.items():
            updated_calls = set()
            for call in details['calls']:
                # Check if the call exists in the mapping
                if call in method_mapping:
                    updated_calls.add(method_mapping[call])
                elif call in functions:  # For global functions, fallback to their own name
                    updated_calls.add(call)
                else:
                    updated_calls.add(call)  # Keep as-is if not found
            details['calls'] = updated_calls

    # Count function/method calls globally
    call_counts = {}
    for details in functions.values():
        for call in details['calls']:
            call_counts[call] = call_counts.get(call, 0) + 1

    # Create a directed graph of the call relationships
    graph = nx.DiGraph()
    for function_name, details in functions.items():
        graph.add_node(function_name, args=details['args'])
        for called_function in details['calls']:
            if call_counts.get(called_function, 0) <= max_func_call:
                graph.add_edge(function_name, called_function)

    #     # Use a hierarchical layout for nodes (top-to-bottom)
    #     root_nodes = [node for node in graph.nodes if graph.in_degree(node) == 0]
    #     levels = {}
    #     for root in root_nodes:
    #         for node, level in nx.single_source_shortest_path_length(graph.reverse(), source=root).items():
    #             levels[node] = max(levels.get(node, 0), level)

    # Use an iterative level refinement algorithm for better hierarchy
    root_nodes = [node for node in graph.nodes if graph.in_degree(node) == 0]
    levels = {node: 5 for node in graph.nodes}  # Initialize all nodes to level 5

    # Iteratively refine node levels
    for _ in range(levels_iters):
        # Step 2: Ensure levels are at least 1 greater than their predecessors
        for node in graph.nodes:
            for pred in graph.predecessors(node):
                levels[node] = max(levels[node], levels[pred] + 1)

        # Step 3: Ensure levels are at least 1 less than the functions they call
        for node in graph.nodes:
            for succ in graph.successors(node):
                levels[node] = min(levels[node], levels[succ] - 1)

    # Normalize levels to fit the plotting space
    min_level, max_level = min(levels.values()), max(levels.values())
    levels = {node: (lvl - min_level) / (max_level - min_level) * (canvas_height - 100) for node, lvl in levels.items()}

    # Distribute nodes while maintaining aspect ratio
    node_positions = {}
    level_y_spacing = 200
    max_nodes_per_row = canvas_width // 300  # Approximate number of nodes per row
    current_y = 0

    for level in sorted(set(levels.values())):
        nodes_in_level = [node for node, lvl in levels.items() if lvl == level]

        # Break nodes into multiple rows if they exceed max_nodes_per_row
        rows = [nodes_in_level[i:i + max_nodes_per_row] for i in range(0, len(nodes_in_level), max_nodes_per_row)]
        for row in rows:
            current_x = -(len(row) - 1) * 300 / 2  # Center nodes in the row
            for node in row:
                node_positions[node] = existing_positions.get(node, (current_x, current_y))
                current_x += 300
            current_y += level_y_spacing

    # Add any nodes missing in `levels` (unconnected nodes)
    x_spacing = 300
    y_spacing = 200
    unpositioned_nodes = [node for node in functions.keys() if node not in node_positions]
    for i, node in enumerate(unpositioned_nodes):
        node_positions[node] = existing_positions.get(node, (
        (i % max_nodes_per_row) * x_spacing, current_y + (i // max_nodes_per_row) * y_spacing))

    # Generate colormap for nodes
    colormap = cmc.romaO
    node_colors = {node: to_hex(colormap(i / len(node_positions))) for i, node in enumerate(node_positions)}

    # Group nodes by class
    group_nodes = []
    for class_name, methods in class_methods.items():
        method_positions = [node_positions[method] for method in methods.keys() if method in node_positions]
        if method_positions:
            x_positions, y_positions = zip(*method_positions)
            x_min, x_max = min(x_positions), max(x_positions)
            y_min, y_max = min(y_positions), max(y_positions)
            group_nodes.append({
                "id": f"group_{class_name}",
                "x": x_min - 50,
                "y": y_min - 50,
                "width": x_max - x_min + 350,
                "height": y_max - y_min + 200,
                "type": "group",
                "label": class_name
            })

    # Generate nodes for the canvas
    nodes = []
    node_ids = {}
    for i, (function_name, (x, y)) in enumerate(node_positions.items()):
        node_id = f"node_{i}"
        nodes.append({
            "id": node_id,
            "x": x,
            "y": y,
            "width": 250,
            "height": 100,
            "type": "text",
            "text": function_name,
            "color": node_colors[function_name]
        })
        node_ids[function_name] = node_id

    # Generate edges for the canvas
    edges = []
    for function_name, details in functions.items():
        print(f'-------------------------------------')
        print(f'\n{function_name} calls :')

        # Loop for checking if there are too many outgoing calls.
        out_call_counts = 0
        for called_function in details['calls']:
            print(f"  Edge: {function_name} -? {called_function}")
            if called_function in node_ids and call_counts.get(called_function, 0) <= max_func_call:
                out_call_counts = out_call_counts + 1
        print(f'{out_call_counts=}')

        if out_call_counts <= max_func_call * 2:
            for called_function in details['calls']:
                print(f"  Edge: {function_name} -? {called_function}")
                if called_function in node_ids and call_counts.get(called_function, 0) <= max_func_call:
                    print(f"  Edge: {function_name} -> {called_function}")
                    edges.append({
                        "id": f"edge_{function_name}_to_{called_function}",
                        "fromNode": node_ids[function_name],
                        "fromSide": "bottom",
                        "toNode": node_ids[called_function],
                        "toSide": "top",
                        "color": node_colors[function_name]
                    })
        else:
            # Adding light gray edges for those that fail to meet the max_func_call criteria
            for called_function in details['calls']:
                print(f"  Edge: {function_name} -? {called_function}")
                if called_function in node_ids and call_counts.get(called_function, 0) <= max_func_call:
                    print(f"  Edge: {function_name} -> {called_function}")
                    edges.append({
                        "id": f"edge_{function_name}_to_{called_function}",
                        "fromNode": node_ids[function_name],
                        "fromSide": "bottom",
                        "toNode": node_ids[called_function],
                        "toSide": "top",
                        "color": "#EEEEEE"
                    })
    #     print('------------------\n')
    #     for edge in edges:
    #         print(f'{edge=}')

    # Sort the edges with color "#EEEEEE" first
    # So that these edges are placed below the other coloured edges.
    edges = sorted(edges, key=lambda edge: edge['color'] != "#EEEEEE")

    #     print('------------SORTING------\n')
    #     for edge in edges:
    #         print(f'{edge=}')

    # Write the canvas JSON file
    canvas_data = {
        "nodes": nodes + group_nodes,
        "edges": edges
    }
    with open(canvas_file_path, 'w') as file:
        json.dump(canvas_data, file, indent=2)

    print(f"Canvas file saved to {canvas_file_path}")
