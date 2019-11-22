from request import get_rooms_html
from parse import get_availability_map
from availability import get_available_rooms

from datetime import datetime

today = datetime.now()

today_html = get_rooms_html(today.isoformat())
availability_map = get_availability_map(today_html)

now_timeslot = (2 * (today.hour - 8)) + (today.minute // 30)

available_rooms = get_available_rooms(
    availability_map, today.weekday(), now_timeslot)

for room, time in available_rooms:
    print("Room {} is available until {}".format(room, time))
