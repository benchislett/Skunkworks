from typing import List, Dict
from bs4 import BeautifulSoup
import datetime

def get_availability_map(date_str: str, room_to_html: Dict[str, str]) -> Dict[str, Dict[str, str]]:
    """Return a dictionary containing mappings from room names
    to a dictionary of available start times to the end of that availability
    period, on day date_str (ISO format).
    """

    for room in room_to_html:
        html = room_to_html[room]
        soup = BeautifulSoup(html, 'html.parser')
        day = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        for row in soup.findAll('tr'):
            contents = row.findAll('td')
            if len(contents) > 3:
                print(contents[1].contents)


