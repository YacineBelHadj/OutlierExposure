from pytz import utc
from datetime import datetime,  timezone

events_name1={
    'Remove bolt':datetime(2022,4,25,8,5,0,tzinfo=utc),
  #  'Incomplete (?) Bolt':datetime(2022,4,27,8,57,0,tzinfo=utc),
    'Loose bolt':datetime(2022,4,29,9,22,0,tzinfo=utc),
    'Tighten bolt':datetime(2022,5,3,8,35,0,tzinfo=utc),
    'Buckling EW916':datetime(2022,5,9,8,5,0,tzinfo=utc),
    'Removal EW916':datetime(2022,5,16,7,36,0,tzinfo=utc),
    'Removal EW918':datetime(2022,5,23,12,4,0,tzinfo=utc),
    'Replacement EW916/918':datetime(2022,5,30,6,51,tzinfo=utc),
    'Grinding EW196':datetime(2022,6,7,8,9,tzinfo=utc),
  #  'climbing start':datetime(2022,6,7,7,59,tzinfo=utc),
  #  'climbing end':datetime(2022,6,7,8,6,tzinfo=utc),        
    'Reinforcement EW916 ':datetime(2022,6,13,8,6,tzinfo=utc), # Exact hour not documented
    'Power line work' :datetime(2022,6,20,20,tzinfo=utc)
}
events_name = {chr(i+97):v for i,v in enumerate(events_name1.values())}


def add_event(ax,events_name=events_name,rotation=0,ha='left',**kwargs):
    #add the events
    for n,dt in events_name.items():
        ax.axvline(dt,linestyle='--',zorder=1,color='red')
    #add the ticks, labels
    xlims = ax.get_xlim() 
    twin_axes=ax.twiny() 
    twin_axes.set_xlim(xlims[0],xlims[1])
    twin_axes.set_xticks(list(events_name.values()))
    twin_axes.set_xticklabels(list(events_name.keys()), rotation = rotation, ha=ha,)
    return ax,twin_axes

def add_event_text(ax,events_name,rotation =45,ha='left',yloc=-1):
    for k,v in events_name.items():
        ax.text(v,yloc,k,rotation=rotation,ha=ha)

def add_event_labels(ax,events_name,rotation=45,ha='left',**kwargs):

    xlims = ax.get_xlim() 
    twin_axes=ax.twiny() 
    twin_axes.set_xlim(xlims[0],xlims[1])
    twin_axes.set_xticks(list(events_name.values()),zorder= 1)
    twin_axes.set_xticklabels(list(events_name.keys()), rotation = rotation, ha=ha, zorder= 1)
    return ax,twin_axes

def add_event_line(ax,events_name,rotation=45,ha='left',**kwargs):
    #add the events

    for n,dt in events_name.items():
        ax.axvline(dt,**kwargs)
    #add the ticks, labels
    return ax
