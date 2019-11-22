def get_available_rooms(availability_map: Dict[str, Dict[int, List[str]]], day: int, timeslot: int) -> List[Tuple[str, str]]:
    """Return a list of available rooms and closing times for a given weekday day,
    timeslot (in UTSC Calendar format) and availability_map
    """
    rooms = []
    for room in availability_map:
        try:
            if availability_map[room][day][timeslot] == 'Empty':
                length = 1
                while availability_map[room][day][timeslot + length] == 'Empty':
                    length += 1
                close_time = str((timeslot + length) // 2 + 8) + str(':00' if (timeslot + length) % 2 == 0 else ':30')
                rooms.append((room, close_time))
        except:
            pass
    return rooms

