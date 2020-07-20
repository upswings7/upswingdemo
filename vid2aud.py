import moviepy.editor
# Replace the parameter with the location of the video
def vidtoaud(file):
	video = moviepy.editor.VideoFileClip(file).subclip(0,50)
	audio = video.audio
	# Replace the parameter with the location along with filename
	audio.write_audiofile("uploads/sample.wav")