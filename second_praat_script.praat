
phoneme_tier = 1
filename$ = "formant_pitch_spreadsheet.csv"
writeFileLine: filename$, "interval,phoneme,F1,F2,F3,Pitch,Duration"

writeInfoLine("Results:")

for i from 1 to 10
	
	selectObject: "TextGrid 10-Daniel"
	start_time = Get start time of interval: phoneme_tier, i
	end_time = Get end time of interval: phoneme_tier, i
	duration = end_time - start_time
	midpoint = start_time + duration/2
	label$ = Get label of interval: 1, i

	if label$ != ""
	
		selectObject: 12


		f1 = Get value at time: 1, midpoint, "hertz", "linear"
		f2 = Get value at time: 2, midpoint, "hertz", "linear"
		f3 = Get value at time: 3, midpoint, "hertz", "linear"

		selectObject: "Pitch 10-Daniel"
		pitch = Get value at time: midpoint, "Hertz", "linear"

		appendInfoLine: i, " ", label$, " ", midpoint
		appendInfoLine: "F1: ", f1
		appendInfoLine: "F2: ", f2
		appendInfoLine: "F3: ", f3
		appendInfoLine: "Pitch: ", pitch
		appendInfoLine: ""

		appendFileLine: filename$, i, ",",
					...label$, ",",
					...f1, ",",
					...f2, ",",
					...f3, ",",
					...pitch, ",",
					...duration

	endif
	
endfor


appendInfoLine("Done")


selectObject: "Sound 10-Daniel"
formant1 = To Formant (burg): 0, 5, 5000, 0.025, 50
