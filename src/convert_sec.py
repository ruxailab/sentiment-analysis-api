import re

# Function to convert milliseconds to HH:MM:SS format
def format_time(milliseconds):
    seconds = milliseconds // 1000
    minutes = seconds // 60
    hours = minutes // 60
    return f"{hours:02}:{minutes % 60:02}:{seconds % 60:02}"

# Function to convert a single line
def convert_line(line):
    # Regular expression to match the timestamp pattern
    match = re.match(r'\[(\d+) - (\d+)\] (Speaker .+): (.+)', line)
    if match:
        start_ms = int(match.group(1))
        end_ms = int(match.group(2))
        speaker = match.group(3)
        text = match.group(4)
        start_time = format_time(start_ms)
        end_time = format_time(end_ms)
        return f"[{start_time} - {end_time}] {speaker}: {text}\n"
    return line

# Read the input file and process each line
with open("utterances_timestamps_transcript.txt", "r") as infile, open("utterances_timestamps_hr_transcript.txt", "w") as outfile:
    for line in infile:
        converted_line = convert_line(line)
        outfile.write(converted_line)

# # Verify the output by reading the file back (optional)
# with open("output_transcript.txt", "r") as file:
#     content = file.read()
#     print(content)
