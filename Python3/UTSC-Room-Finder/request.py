from typing import List, Dict
import requests
import json

BUILDING_TO_ROOMS = {'AA': ['112', '204', '205', '206', '207', '208', '209', '223', '332', '334'],
                     'BV': ['260', '335', '359', '361', '363'],
                     'HL': ['B101', 'B106', 'B108', 'B110'],
                     'HW': ['214', '215', '216', '308', '402', '408'],
                     'IC': ['120', '130', '200', '204', '208', '212', '220', '230', '300', '302', '320', '326', '328'],
                     'MW': ['110', '120', '160', '170', '223', '262', '264'],
                     'SW': ['128', '143', '309', '319'],
                     'SY': ['110']
                     }

LINK = 'https://intranet.utsc.utoronto.ca/intranet2/RegistrarService?'


def make_room_list(buildings: List[str]) -> List[str]:
    """Return the acceptable rooms given a list of acceptable buildings.
    If buildings is None, use all rooms.
    """

    if buildings is None:
        buildings = BUILDING_TO_ROOMS.keys()

    rooms = []
    for building in BUILDING_TO_ROOMS:
        if building in buildings:
            for room_number in BUILDING_TO_ROOMS[building]:
                rooms.append(building + '-' + room_number)
    return rooms


def get_rooms_html(date_str: str, buildings: List[str] = None) -> Dict[str, str]:
    """Return a dictionary containing mappings from room names
    to a string of an html table for the room availability on a
    given date in ISO standard YYYY-MM-DD format, date_str, given
    possible building codes buildings.
    """

    rooms = make_room_list(buildings)
    room_str = 'room=' + ','.join(rooms)

    date_str = 'day=' + date_str

    req_str = '&'.join([LINK, room_str, date_str])
    r = requests.get(req_str)
    # This small application can simply let request errors bubble up to the user

    return r.json()
