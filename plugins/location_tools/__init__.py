from dataclasses import dataclass

from plugins.location_tools import locationtools
from plugins.requirement import PluginRequirement #, Plugin, 


@dataclass
class LocationToolsRequirement(PluginRequirement):
    name: str = 'location_tools'
    documentation: str = locationtools.DOCUMENTATION


# class LocationToolsPlugin(Plugin):
#     name: str = 'location_tools'
