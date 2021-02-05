# pypl2_example.py - PyPL2 usage example
#
# (c) 2016 Plexon, Inc., Dallas, Texas
# www.plexon.com - support@plexon.com
# 
# This software is provided as-is, without any warranty.
# You are free to modify or share this file, provided that the above
# copyright notice is kept intact.

from pypl2 import pl2_ad, pl2_spikes, pl2_events, pl2_info
import sys
import os

def choose_file():
    try:
        import tkinter
    except ImportError:
        print("Tkinter not installed.")
        exit()
        
    from tkinter import filedialog
    
    #Suppress the Tkinter root window
    tkroot = tkinter.Tk()
    tkroot.withdraw()
    
    return str(tkinter.filedialog.askopenfilename())
    
if __name__ == "__main__":
    
    #If no file is passed at the command line, or if the file
    #passed can not be found, open a file chooser window.
    if len(sys.argv) < 2:
        filename = os.path.abspath(choose_file())
    else:
        filename = os.path.abspath(sys.argv[1])
        if not os.path.isfile(filename):
            filename = choose_file()
    
    ######################
    # pl2_info() Example #
    ######################
    
    #Get file info and print out interesting information
    spkinfo, evtinfo, adinfo = pl2_info(filename)

    print("\nContinuous A/D Channel Info from pl2_info()")
    print("\n# Channel Name\tCount")
    print("- ------------\t-----")
    for n in range(len(adinfo)):
        print("{} {}\t{}".format(adinfo[n].channel, adinfo[n].name, adinfo[n].n))
    
    print("\nSpike Channel Info from pl2_info()")
    print("\n# Channel Name\tUnit A\tUnit B\tUnit C\tUnit D\tUnsorted")
    print("- ------------\t------\t------\t------\t------\t--------")
    for n in range(len(spkinfo)):
        print("{} {}\t{}\t{}\t{}\t{}\t{}".format(spkinfo[n].channel,
                                                   spkinfo[n].name,
                                                   spkinfo[n].units[1],
                                                   spkinfo[n].units[2],
                                                   spkinfo[n].units[3],
                                                   spkinfo[n].units[4],
                                                   spkinfo[n].units[0]))
                                                   
    print("\nEvent Channel Info from pl2_info()")
    print("\n# Channel Name\tCount")
    print("- ------------\t-----")
    for n in range(len(evtinfo)):
        print("{} {}\t{}".format(evtinfo[n].channel, evtinfo[n].name, evtinfo[n].n))
    
    ###########################################
    # pl2_ad, pl2_spikes, pl2_events Examples #
    ###########################################
    
    #Get continuous a/d data on first channel and print out interesting information
    ad = pl2_ad(filename, 0)
    print("\nContinuous A/D Channel 0 Data from pl2_ad()")
    print("\nFrequency Number of Points First Four A/D Points (mV)")
    print("--------- ---------------- ---------------------")
    print("{:<10}{:<17}{}, {}, {}, {}".format(int(ad.adfrequency),
                                        ad.n,
                                        ad.ad[0] * 1000,
                                        ad.ad[1] * 1000,
                                        ad.ad[2] * 1000,
                                        ad.ad[3] * 1000))
    
    #Get spikes on first channel and print out interesting information on the first four spikes.
    spikes = pl2_spikes(filename, 0)
    print("\nSpike Channel 0 Data for First Four Waveforms from pl2_spikes()")
    print("\nTimestamps (s) Unit First Four Waveform Points (uV)")
    print("-------------- ---- -------------------------------")
    for n in range(4):
        print("{:<15}{:<5}{}, {}, {}, {}".format(spikes.timestamps[n],
                                           spikes.units[n],
                                           spikes.waveforms[n][0] * 1000000,
                                           spikes.waveforms[n][1] * 1000000,
                                           spikes.waveforms[n][2] * 1000000,
                                           spikes.waveforms[n][3] * 1000000))
    
    #Get event data on select channels and print out interesting information
    print("\nEvent Data from pl2_events()")
    print("\nEvent   Number of Events First Timestamp (s)")
    print("------- ---------------- -------------------")

    for n in range(len(evtinfo)):
        evt = pl2_events(filename, evtinfo[n].name)
        print("{:<7} {:<16} {}".format(evtinfo[n].name, evt.n, evt.timestamps[0]))

    #Get strobed event data and print out interesting information
    strobedevt = pl2_events(filename, 'Strobed')
    print("\nFirst Ten Strobe Values from pl2_events()")
    print("\nStrobe Value Timestamp (s)")
    print("------------ -------------") 
    for n in range(10):
        print("{:<12} {}".format(strobedevt.values[n], strobedevt.timestamps[n]))
    
    