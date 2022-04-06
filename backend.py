import pandas

def get_supply_chain():
    supplyChain = pandas.read_csv('supplyChain.csv')
    return supplyChain

def get_events():
    events = pandas.read_csv('events.csv')
    return events