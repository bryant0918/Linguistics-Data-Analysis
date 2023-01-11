# Praat Script
phoneme_tier = 2
filename$ = "intensity_pitch_spreadsheet.csv"
writeFileLine: filename$, "Interval,phoneme,Intensity,Pitch,Start Time,End Time"

selectObject: "TextGrid 1_ne_1_1_en"
intervals = Get number of intervals: 2

writeInfoLine("Results:")

for i from 1 to intervals
	
	selectObject: "TextGrid 1_ne_1_1_en"
	start_time = Get start time of interval: phoneme_tier, i
	end_time = Get end time of interval: phoneme_tier, i
	label$ = Get label of interval: phoneme_tier, i
	
	# True for all stressed vowels (primary and secondary)
	bool = endsWith(label$, "1") or endsWith(label$, "2")

	if label$ != "" and bool == 1
		
		# Select Intensity Object to get mean Intensity
		selectObject: "Intensity 2015-11-0010-1-nephi-01-male-voice-64k-eng"
		intensity = Get mean: start_time, end_time, "energy"


		# Select Pitch Object to get mean Pitch
		selectObject: "Pitch 2015-11-0010-1-nephi-01-male-voice-64k-eng"
		pitch = Get mean: start_time, end_time, "Hertz"
		
		# Do not write any undefined values
		if pitch != pitch+1 and intensity != intensity+1
			# Write to CSV file
			appendFileLine: filename$, i, ",",
					...label$, ",",
					...intensity, ",",
					...pitch, ",",
					...start_time, ",",
					...end_time
		endif

	endif
	
endfor


appendInfoLine("Done")
