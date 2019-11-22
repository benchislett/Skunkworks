from typing import List, Dict
from bs4 import BeautifulSoup
import datetime

def get_availability_map(room_to_html: Dict[str, str]) -> Dict[str, Dict[str, str]]:
    """Return a dictionary containing mappings from room names
    to a dictionary of days to list of strings representing availability at given times,
    from 8:00 to 21:00 inclusive, from the html table room_to_html.
    Availability will either be "Empty", or the listed event.
    """
    out_map = {}
    for room in room_to_html:
        out_map[room] = {}

        html = room_to_html[room]
        soup = BeautifulSoup(html, 'html.parser')

        for day in range(7):
            out_map[room][day] = []
            timeout = 0
            current_event = None
            for row in soup.findAll('tr'):
                contents = row.findAll('td')
                if contents is None or len(contents) < 2:
                    continue
                if timeout > 0:
                    timeout -= 1
                    out_map[room][day].append(current_event)
                else:
                    item = contents[1].extract()
                    print(item)
                    if item.has_attr('rowspan') and item['rowspan'] is not None:
                        print(item['rowspan'])
                        timeout = int(item['rowspan']) - 1
                        current_event = item.get_text(separator=" ")
                        out_map[room][day].append(current_event)
                    else:
                        out_map[room][day].append('Empty')

    return out_map
