import streamlit as st
from streamlit_timeline import st_timeline
from datetime import datetime, timedelta 
from timeline import load_hdf5, load_vocab, process_timeline_events, get_patient_timeline
import re
from collections import defaultdict
from streamlit_tags import st_tags

st.set_page_config(layout="wide")

# Base date for the timeline (can be modified)
base_date = datetime(2022, 10, 1)
# Initialize session state for date tracking
if "current_date" not in st.session_state:
    st.session_state.current_date = base_date

# File paths
HDF5_FILE = "/home/shtlp_0094/workspace/ethos-paper/ethos/data/tokenized_datasets/mimic_test_timelines_p10.hdf5"
VOCAB_FILE = "/home/shtlp_0094/workspace/ethos-paper/ethos/data/tokenized_datasets/mimic_vocab_t763.pkl"

# Load data
data = load_hdf5(HDF5_FILE)
itos = load_vocab(VOCAB_FILE)


# Extract `times` and `tokens` from data
times = data["times"]
tokens = data["tokens"]

# Function to generate color-coded timeline for Streamlit
def categorize_input(input_str):
    if input_str.startswith("LAB_"):
        return "LAB"
    elif input_str.startswith("ICD_PCS"):
        return "ICD PCS"
    elif input_str.startswith("ICD_"):
        return "ICD CM"
    elif "SOFA" in input_str:
        return "SOFA"
    elif "BLOOD_PRESSURE" in input_str:
        return "LAB"
    elif re.search(r"_(\d+)([mhdw]|mt)-(\d+)([mhdw]|mt)", input_str) or re.search(r"_(\d+)([mhdw]|mt)", input_str):
        return "time"
    elif input_str.startswith("ATC_"):
        return "ATC"
    else:
        return "General"


def decode(token_list):
    return [itos[token] if token in itos else f"UNKNOWN_{token}" for token in token_list]

def replace_underscores(input_string: str) -> str:
    """
    Replaces all underscores in the input string with spaces.
    :param input_string: The string to modify.
    :return: The modified string with underscores replaced by spaces.
    """
    return input_string.replace("_", " ")


# Get timeline for a specific patient
patient_id = 10039997
patient_timeline = get_patient_timeline(patient_id, data, decode)

# # Display the timeline if found
if patient_timeline is not None:
    print(f"Timeline for Patient {patient_id}:\n")
    print(patient_timeline.head())


data = [tuple(row) for row in patient_timeline.itertuples(index=False, name=None)]


# Merge consecutive LAB events
merged_events = []
i = 0

while i < len(data):
        event = data[i]  # Access tuple
        if event[0].startswith("LAB") and i + 1 < len(data):  # event[0] is the Event Name
            next_event = data[i + 1]
            
            # Merge LAB event with the next event
            merged_event = (
                f"{event[0]} ({next_event[0]})",  # Merge event names
                event[1],  # Token Mapping remains unchanged
                event[2],  # Merge Time Passed (Days)
                event[3] ,  # Merge Time Passed (Hours)
                event[4]    # Merge Time Passed (Minutes)
            )
            
            data[i] = merged_event  # Update current event
            del data[i+1]  # Remove next event
            i += 1
        elif "BLOOD_PRESSURE" in event[0] and i + 2 < len(data):
                previous_events = [data[i+1],data[i+2]]
                merged_event = (
                    f"{event[0]}({previous_events[0][0]}{previous_events[1][0]})",  # Merge event names
                    event[1],  # Keep Token Mapping from two events before
                    event[2],  # Keep Time Passed (Days)
                    event[3],  # Keep Time Passed (Hours)
                    event[4]   # Keep Time Passed (Minutes)
                )
     
                data[i] = merged_event  # Update BLOOD PRESSURE event with merged data
                del data[i+1]  # Remove event two places before
                del data[i+1]  # Remove event one place before (index shifts after deletion)
                i += 1
        else:
            i += 1


# Convert data to timeline format
items = []
for idx, (event, token, days, hours, minutes ) in enumerate(data):
    event_time = base_date + timedelta(days=days, hours=hours, minutes=minutes)
    content = replace_underscores(event)
    content_group = categorize_input(event) 
    if content_group == "LAB":
        items.append({"id": idx + 1, "content": content,"group": content_group, "start": event_time.strftime("%Y-%m-%d %H:%M:%S"), "title": event, "style": "background-color: yellow"})
    elif content_group == "ICD CM":
        items.append({"id": idx + 1, "content": content,"group": content_group,"start": event_time.strftime("%Y-%m-%d %H:%M:%S"), "title": event, "style": "background-color: lightgreen"})
    elif content_group == "SOFA":
        items.append({"id": idx + 1, "content": content,"group": content_group, "start": event_time.strftime("%Y-%m-%d %H:%M:%S"), "title": event, "style": "background-color: lightblue"})
    elif content_group == "ICD PCS":
        items.append({"id": idx + 1, "content": content,"group": content_group, "start": event_time.strftime("%Y-%m-%d %H:%M:%S"), "title": event, "style": "background-color: orange"})
    elif content_group == "time":
        # Default end time is start time
        end_time = event_time 
        # Check if there's a next event
        if idx + 1 < len(data):
            next_event_time = base_date + timedelta(days=data[idx + 1][2], hours=data[idx + 1][3], minutes=data[idx + 1][4])
            end_time = next_event_time
        else:
            # If no next event, estimate end time (using lower bound of the detected time token)
            match = re.search(r"_(\d+)([mhdw]|mt)", event)  # Extract numeric value and unit
            if match:
                value, unit = int(match.group(1)), match.group(2)
                if unit == "m":  # minutes
                    end_time = event_time + timedelta(minutes=value)
                elif unit == "h":  # hours
                    end_time = event_time + timedelta(hours=value)
                elif unit in ["d", "w"]:  # days or weeks
                    end_time = event_time + timedelta(days=value if unit == "d" else value * 7)
        items.append({"id": idx + 1, "content": content,"group": content_group, "start": event_time.strftime("%Y-%m-%d %H:%M:%S"),"end": end_time.strftime("%Y-%m-%d %H:%M:%S") , "title": event, "style": "background-color: lightgrey"})
    elif content_group == "ATC":
        items.append({"id": idx + 1, "content": content,"group": content_group, "start": event_time.strftime("%Y-%m-%d %H:%M:%S"), "title": event, "style": "background-color: light blue"})
    else:
        items.append({"id": idx + 1, "content": content,"group": content_group, "start": event_time.strftime("%Y-%m-%d %H:%M:%S"), "title": event, "style": "background-color: pink"})


# Display timeline
st.subheader("Patient Health Timeline")
toggle_state = st.toggle("Timeline View Mode")
if not toggle_state:
    st.markdown("#### **Select View Mode:**")  # Increases font size and makes it bold
    view_option = st.radio("", ["Daily View", "Weekly View", "Monthly View","Custom Range"], index=0, horizontal= True)
    # Calculate date range based on the selected view
    if view_option == "Daily View":
        start_date = st.session_state.current_date.replace(hour=0, minute=0, second=0)  # Start of the day
        end_date = st.session_state.current_date.replace(hour=23, minute=59, second=59)  # End of the day
    elif view_option == "Weekly View":
        start_date = st.session_state.current_date - timedelta(days=st.session_state.current_date.weekday())  # Monday of the week
        end_date = start_date + timedelta(days=6, hours=23, minutes=59, seconds=59)  # Sunday end of day
    elif view_option == "Monthly View":
        start_date = st.session_state.current_date.replace(day=1)  # First day of the month
        next_month = st.session_state.current_date.replace(day=28) + timedelta(days=4)  # Jump to next month
        end_date = next_month.replace(day=1) - timedelta(seconds=1)  # Last day of the month
    elif view_option == "Custom Range":
        # User selects start and end date with constraints
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", value=st.session_state.current_date.date(), min_value=base_date.date(), max_value=datetime.today().date())

        with col2:
            end_date = st.date_input("End Date", value=start_date, min_value=start_date, max_value=datetime.today().date())

        # Convert date to datetime format for processing
        start_date = datetime.combine(start_date, datetime.min.time())
        end_date = datetime.combine(end_date, datetime.max.time())

    # Extract unique event groups, excluding "time"
    event_groups = sorted(set(item["group"] for item in items ))

    st.sidebar.markdown('<p style="font-size:20px; font-weight:bold; margin-bottom:-5px;">Select Events:</p>', unsafe_allow_html=True)
    # Create checkboxes for each event
    # Initialize session state for event checkboxes
    if "event_checkboxes" not in st.session_state:
        st.session_state.event_checkboxes = {event: True for event in event_groups}  # Default all to checked
    selected_events = []
    with st.sidebar.expander("Choose Events", expanded=True):
        all_selected = st.sidebar.checkbox("All Events", value=True, key="all_events")  # Select all by default
        
        # Update session state when "All Events" is toggled
        if all_selected:
            for event in event_groups:
                st.session_state.event_checkboxes[event] = True
        else:
            # If all events were previously checked, uncheck them (allows manual selection)
            if all(event_checked for event_checked in st.session_state.event_checkboxes.values()):
                for event in event_groups:
                    st.session_state.event_checkboxes[event] = False

        event_checkboxes = {}
        for event in event_groups:
            event_checkboxes[event] = st.checkbox(event, value=st.session_state.event_checkboxes[event])

        # Update selected_events based on checkboxes
        if all_selected:
            selected_events = event_groups  # Select all if "All Events" is checked
        else:
            selected_events = [event for event, checked in event_checkboxes.items() if checked]
    # Adjust the time frame based on selected event type
    def fit_to_selected_events():
        grouped_events = {}

        for item in items:
            event_date = datetime.strptime(item["start"], "%Y-%m-%d %H:%M:%S")

            if start_date <= event_date < end_date:
                group = item["group"]
                if group in selected_events:
                    if group not in grouped_events:
                        grouped_events[group] = []
                    grouped_events[group].append(event_date)

        if grouped_events:
            # Get min & max dates of selected event(s)
            new_start_date = min(min(dates) for dates in grouped_events.values())
            new_end_date = max(max(dates) for dates in grouped_events.values())

            # Update session state and rerun
            st.session_state.current_date = new_start_date
            st.session_state.end_date = new_end_date  # Ensure end_date updates too
            st.rerun()  # 🔄 Refresh Streamlit app to reflect changes
        else:
            st.warning(f"**No events found for {', '.join(selected_events)} in the selected window.**")

    if not toggle_state:
        if st.sidebar.button("### Apply to Selected Events"):
            fit_to_selected_events()
    # Navigation buttons to move between weeks or months
    if not toggle_state:
        st.markdown("""
            <style>
            div.stButton > button {
                font-size: 35px !important; /* Increases font size */
                font-weight: bold !important; /* Makes text bold */
                padding: 10px 10px !important; /* Adds padding for better appearance */
            }
            </style>
            """, unsafe_allow_html=True)
        col1, col2 = st.columns(2)

        with col1:
            if st.button("**◀ Previous**"):
                if view_option == "Daily View":
                    st.session_state.current_date -= timedelta(days=1)
                elif view_option == "Weekly View":
                    st.session_state.current_date -= timedelta(weeks=1)
                elif view_option == "Monthly View":
                    st.session_state.current_date -= timedelta(days=st.session_state.current_date.day)  # Move to the previous month
                st.rerun()

        with col2:
            if st.button("**Next ▶**"):
                if view_option == "Daily View":
                    st.session_state.current_date += timedelta(days=1)
                elif view_option == "Weekly View":
                    st.session_state.current_date += timedelta(weeks=1)
                elif view_option == "Monthly View":
                    next_month = st.session_state.current_date.replace(day=28) + timedelta(days=4)  # Jump to next month
                    st.session_state.current_date = next_month.replace(day=1)  # Move to the next month
                st.rerun()
    content_group =  set(categorize_input(event) for event, _, _, _, _ in data)
    groups = [{"id": group, "content": group} for group in content_group if group in selected_events]
    # Filter events within the selected date range
    filtered_items = [
        item for item in items
        if start_date <= datetime.strptime(item["start"], "%Y-%m-%d %H:%M:%S") <= end_date
        and (item["group"] in selected_events)
    ]

    st.write(f"### Events from {start_date.strftime('%d %b %Y')} to {end_date.strftime('%d %b %Y')}")



clustered_events = defaultdict(lambda: defaultdict(list))



def get_cluster_color(content_group):
    """Returns the color for each content group cluster."""
    return {
        "LAB": "yellow",
        "ICD CM": "lightgreen",
        "ATC": "lightblue",    
        "time": "lightgrey",
        "ICD PCS": "orange",
        "SOFA": "red"      
    }.get(content_group, "pink")  # Default is orange if unknown

def estimate_end_time(event_time, event_name):
    """Estimates the end time for 'time' group events based on detected duration tokens."""
    match = re.search(r"_(\d+)([mhdw]|mt)", event_name)  # Extract numeric value and unit
    if match:
        value, unit = int(match.group(1)), match.group(2)
        if unit == "m":  # minutes
            return event_time + timedelta(minutes=value)
        elif unit == "h":  # hours
            return event_time + timedelta(hours=value)
        elif unit in ["d", "w"]:  # days or weeks
            return event_time + timedelta(days=value if unit == "d" else value * 7)
    return event_time  # Default to same start time if no duration found


def create_timeline_item(event_id, event_name, content_group, timestamp, end_time=None):
    """Creates a timeline item for Streamlit timeline visualization."""
    style_color = get_cluster_color(content_group)
    item = {
        "id": event_id,
        "content": replace_underscores(event_name),
        "group": content_group,
        "start": timestamp,
        "title": event_name,
        "style": f"background-color: {style_color}"
    }
    if end_time:
        item["end"] = end_time.strftime("%Y-%m-%d %H:%M:%S")  # Include end time for time-based events
    return item

for idx, (event, token, days, hours, minutes ) in enumerate(data):
    event_time = base_date + timedelta(days=days, hours=hours, minutes=minutes)
    event_time_str = event_time.strftime("%Y-%m-%d %H:%M:%S")
    content = replace_underscores(event)
    content_group = categorize_input(event)
    # Ensure dictionary structure before appending
    clustered_events[event_time_str][content_group].append((idx + 1, event))

items = []
for timestamp, category_events in clustered_events.items():
    for content_group, events in category_events.items():
        if len(events) > 1 and content_group!= "General":  # Cluster multiple events of the same category
            event_titles = ", ".join([event[1] for event in events])  # Combine event names
            items.append({
                "id": events[0][0],  
                "content": f"{content_group}: {len(events)} events",  # Show category-based clustering
                "group": content_group,
                "start": timestamp,
                "title": event_titles,
                "style": f"background-color: {get_cluster_color(content_group)}"
            })
        else:  # Handle individual events
            event_id, event_name = events[0]
            event_time = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")  # Convert to datetime object

            if content_group == "time":
                idx = event_id - 1  # Adjust index for `data`
                end_time = None

                # Check if there's a next event to determine end time
                if idx + 1 < len(data):
                    next_event_time = base_date + timedelta(
                        days=data[idx + 1][2], hours=data[idx + 1][3], minutes=data[idx + 1][4]
                    )
                    end_time = next_event_time
                else:
                    end_time = estimate_end_time(event_time, event_name)

                items.append(create_timeline_item(event_id, event_name, content_group, timestamp, end_time))
            else:
                items.append(create_timeline_item(event_id, event_name, content_group, timestamp))

# Timeline Options
timeline_options = {
    "min":'08-01-2022', 
    "max": '12-05-2023',
    "editable": False,
    "cluster": True,
    "fitOnDoubleClick": True,
    
}


if toggle_state:
    st.subheader("Patient Health Timeline Summary")
    timeline = st_timeline(items, groups=[], options=timeline_options, height="500px")
    st.subheader("Selected Event Details")
    st.write(timeline)
else :
    st.subheader("Detailed Patient Health Timeline")
    timeline = st_timeline(filtered_items, groups=groups, options={"min": (start_date - timedelta(days=30)).strftime('%Y-%m-%d'),  # Allows scrolling back 30 days
                "max": end_date.strftime('%Y-%m-%d'),    # Allows scrolling forward 30 days
                "editable": False, "verticalScroll": True
                }, height="750px")
    st.subheader("Selected Event Details")
    st.write(timeline)