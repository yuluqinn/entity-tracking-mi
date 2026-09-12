"""
Some helper functions for intermidiate state evaluations, tracing and annotations.
"""
import re
import numpy as np


def detect_removals(S):
    """Detect removals of objects from the world state."""
    num_steps, num_boxes, num_objs = S.shape
    removed_objects = []

    for o in range(num_objs):
        presence_over_time = S[:, :, o].any(axis=1)  # shape: (num_steps,)

        if not presence_over_time.any():
            continue
        last_presence_time = np.where(presence_over_time)[0][-1]
        if last_presence_time < num_steps - 1:
            if not presence_over_time[last_presence_time + 1:].any():
                removed_objects.append(o)
    
    return removed_objects


def detect_local_removals(S, box_id):
    """
    Detect Local Removals of an object from a specific box.
    Need to validate whether the output contains the global removed objects: all global removed objects are also local removed objects, but not vice versa.
    """
    num_steps, num_boxes, num_objs = S.shape
    removed_objects = []

    box_states = S[:, box_id, :]  # shape: (T, NUM_OBJ)

    
    ever_present = np.any(box_states, axis=0)  # shape: (NUM_OBJ,)

    
    final_state = box_states[-1, :] == 0  # shape: (NUM_OBJ,)

    
    removed_mask = np.logical_and(ever_present, final_state)  # shape: (NUM_OBJ,)

    removed_indices = np.where(removed_mask)[0].tolist()

    return removed_indices


def generate_state_matrix(context, object_map, num_boxes=7, num_obj=100, contains_query=True, box_base=0):
    # box_base: lowest box number in the dataset (0 for the original data, 1 for vlm-data);
    # maps literal box numbers to 0-based state-matrix indices
    """
    input:
        context: the prefix context string, consisting of a initial state description, a sequence of operations, and a query sentence.
    Generate a state matrix from the context string.
    The state matrix has shape (num_ops + 1, num_boxes, num_obj),
    """
    seq = [st.strip() for st in context.split('.') if st] 
    ori_state = seq[0]
    if contains_query:
        query = seq[-1]
        op_seq = seq[1:-1] # Might be a problem -- no query now
        
    else:
        op_seq = seq[1:]
        query = None
    state = np.zeros((len(op_seq) + 1, num_boxes, num_obj))
    # initialize world state # not compatible with altform, maybe need to add another branch to parse the altform initial state
    is_altform = False
    # Non-altform:  Box 0 contains xxx; altform: The xxx is in Box 0 or The xxx and yyy are in Box 0 
    if ori_state.startswith("The"): # Super hacky way though, but should keep it working for now.
        is_altform = True
    ori_state_contents = [s.split('contains')[-1].strip().replace("the ", "").split(" and ") for s in ori_state.split(',')] if not is_altform else [
        re.split(r'(?: is|are) in', s)[0].strip().replace("the ", "").split(" and ") for s in ori_state.lower().split(',')] # should work now
    
    for b in range(num_boxes):
        # pasrse the original state
        contents = ori_state_contents[b]
        for c in contents:
            if c in object_map:
                oidx = object_map[c]
                state[0, b, oidx] = 1
    for i, op in enumerate(op_seq):
        timestep = i + 1
        # find the operator defined in the ops, e.g., "Put", "Remove", "Move"
        operator = op.strip().split()[0]
        # there might be two Boxes in the operation, e.g., "Move the map from Box 6 to Box 2"
        box_ids = re.findall(r'Box (\d+)', op)
        if len(box_ids) == 0:
            print("No box id found in operation:", op)
            continue
        elif len(box_ids) == 1:
            # for both Put and Remove, there is only one box id, we use the NUM_BOXES + 1 as the world state for "outside the boxes"
            if operator == "Put":
                src = num_boxes
                tgt = int(box_ids[0]) - box_base
            elif operator == "Remove":
                src = int(box_ids[0]) - box_base
                tgt = num_boxes
        else:
            # only for move from b1 to b2, so b1 is the src, b2 is the tgt
            src = int(box_ids[0]) - box_base
            tgt = int(box_ids[1]) - box_base

        objects = re.findall(r'the ([^ ,.]+)', op)
        obj_idxs = [object_map[o] for o in objects if o in object_map]
        # copy the previous world state
        state[timestep] = state[timestep - 1].copy()
        if operator == "Put":
            for oidx in obj_idxs:
                if state[timestep, tgt, oidx] == 1:
                    print(f"Warning: object {oidx} already in box {src} when putting")
                state[timestep, tgt, oidx] = 1
        elif operator == "Remove":
            for oidx in obj_idxs:
                if state[timestep, src, oidx] == 0:
                    print(f"Warning: object {oidx} not in box {src} when removing")
                state[timestep, src, oidx] = 0
        elif operator == "Move":
            for oidx in obj_idxs:
                if state[timestep, src, oidx] == 0:
                    print(f"Warning: object {oidx} not in box {src} when moving")
                state[timestep, src, oidx] = 0
                state[timestep, tgt, oidx] = 1
        else:
            print("Unknown operator:", operator)
        # print("World state at timestep", timestep)
        # for b in range(num_boxes):
        #     box_contents = [object_list[i] for i in range(num_obj) if state[timestep, b, i] == 1]
        #     print(f"Box {b}: {box_contents}")
    global_removed_objects = detect_removals(state) # now we're interested in the once mentioned but later removed objects. Just implement here for now.
    box_id = int(query.split('Box ')[-1].split()[0]) - box_base if contains_query else None # extract the box id from the query, e.g., "What objects are in Box 2?"
    local_removed_objects = detect_local_removals(state, box_id) if contains_query else None # local removed objects from each box, we can also use this to validate the global removed objects.
        
    return state, global_removed_objects, local_removed_objects
 