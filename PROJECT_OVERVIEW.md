Problem: Singing synthesis models currently have severe issues with
Conforming to the given rhythm
Incorrect or drifting pitch from given score
Lack of controllability from the user’s end; the model generates its own prosody, dynamics etc

Solution: VocaloFlow
A conditional flow matching model: Vocaloid mel-spectrogram (low quality prior) -> human mel-spectrogram(high quality post)
Conditioned on F0 curve of human audio
Mel-spectrogram: a lower-dimensional representation of audio, which is then converted back into audio by a vocoder model like HiFiGan. Soulx-singer has a built-in vocoder model.
Input: A Vocaloid/OpenUTAU singing score, which produces:
A vocaloid singing mel-spectrogram
an F0 prosody curve
Output: Human-sounding singing mel-spectrogram


Data: DALI dataset
Includes annotations for lyrics, notes for approximately 190 hours of English singing
Audio itself is sourced from YouTube




Data Synthesis:
Break each song into n-line sections (lines are defined by DALI)
A single line is around 2-5 seconds of audio
Retrieve note and lyric data for this segment
Generate the post “human” audio from a large singing synthesis model “Soulx-singer”
This generation needs a voice/style prompt for zero-shot generation; we use the same prompt to generate all samples
Soulx-singer also gives the mel-spectrogram from before its vocoder step
Extract note timings/pitches from post audio
Generate the prior “Vocaloid” audio using custom API from extracted notes
Mel-spectrogram must be extracted
(maybe do if this is insufficient:) Use Dynamic Time Warping (DTW) to time-align the vocaloid prior to the posterior
Extract F0 from post audio

Training Data summary:
- mel spectrogram of the final Teto Prior "aligned_mel.npy" with mel settings that match soulx-singer
- mel spectrogram of the soulx-singer target "target_mel.npy"
- extracted f0 curve from target
- (final iteration) notes and lyrics with time stamps (these need to be converted into a phoneme identity mask)
- alignment score data for filtering data


Flow Matching with straight-line optimal transport
Fine tune using existing structure
Match prior mel -> post mel, given F0
Architecture: diffusion transformer
Use overlapping windows for long generation
We can try for one-step or few-step FM by treating this as a ReFlow task


