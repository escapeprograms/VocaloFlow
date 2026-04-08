def get_chunks(mode, notes_annot, words_annot, lines_annot, paragraphs_annot, n_lines=4):
    chunks = [] # each chunk is a list of note indices
    chunk_start_times = []
    chunk_names = []

    if mode == "test":
        for line_idx, line_info in enumerate(lines_annot):
            line_notes = []
            for idx, note_info in enumerate(notes_annot):
                if words_annot[note_info['index']]['index'] == line_idx:
                    line_notes.append(idx)
            if line_notes:
                chunks.append(line_notes)
                chunk_start_times.append(line_info['time'][0])
                chunk_names.append("line_test")
                break

    elif mode == "paragraph":
        for para_idx, para_info in enumerate(paragraphs_annot):
            para_notes = []
            for idx, note_info in enumerate(notes_annot):
                line_idx = words_annot[note_info['index']]['index']
                if lines_annot[line_idx]['index'] == para_idx:
                    para_notes.append(idx)
            if para_notes:
                chunks.append(para_notes)
                chunk_start_times.append(para_info['time'][0])
                chunk_names.append(f"paragraph_{para_idx}")

    elif mode == "line":
        for line_idx, line_info in enumerate(lines_annot):
            line_notes = []
            for idx, note_info in enumerate(notes_annot):
                if words_annot[note_info['index']]['index'] == line_idx:
                    line_notes.append(idx)
            if line_notes:
                chunks.append(line_notes)
                chunk_start_times.append(line_info['time'][0])
                chunk_names.append(f"line_{line_idx}")

    elif mode == "n-line":
        chunk_idx = 0
        for para_idx, para_info in enumerate(paragraphs_annot):
            para_lines = [line_idx for line_idx, line_info in enumerate(lines_annot) if line_info['index'] == para_idx]
            
            for i in range(0, len(para_lines), n_lines):
                group_lines = para_lines[i:i+n_lines]
                group_notes = []
                for idx, note_info in enumerate(notes_annot):
                    if words_annot[note_info['index']]['index'] in group_lines:
                        group_notes.append(idx)
                
                if group_notes:
                    chunks.append(group_notes)
                    chunk_start_times.append(lines_annot[group_lines[0]]['time'][0])
                    chunk_names.append(f"chunk_{chunk_idx}")
                    chunk_idx += 1

    return chunks, chunk_start_times, chunk_names
