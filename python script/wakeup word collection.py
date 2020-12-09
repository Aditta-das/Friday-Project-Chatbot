import pyaudio
import wave, argparse, time, os

class Recorder:
	def __init__(self, args, chunk=1024, sample_format=pyaudio.paInt16, channel=2, fs=44100):
		self.chunk = chunk
		self.sample_format = sample_format
		self.channel = channel
		self.fs = fs
		self.second = args.second
		self.p = pyaudio.PyAudio() # Create an interface to PortAudio
		self.stream = self.p.open(format=self.sample_format,
							                channels=self.channel,
							                rate=self.fs,
							                frames_per_buffer=self.chunk,
							                input=True)

	def recording(self, filename, frames):
		print('Save File {}'.format(filename))
		self.stream.stop_stream()
		self.stream.close()
		# Terminate the PortAudio interface
		self.p.terminate()
		wf = wave.open(filename, 'wb')
		wf.setnchannels(self.channel)
		wf.setsampwidth(self.p.get_sample_size(self.sample_format))
		wf.setframerate(self.fs)
		wf.writeframes(b''.join(frames))
		wf.close()


def record(args):
	index_no = 0
	try:
		while True:
			listener = Recorder(args)
			frames = []
			print("Start Recording...")
			input('press enter to continue. the recoding will be {} seconds. press ctrl + c to exit'.format(args.second))
			time.sleep(0.2)
			for i in range(0, int(listener.fs / listener.chunk * listener.second)):
				data = listener.stream.read(listener.chunk, exception_on_overflow=False)
				frames.append(data)

			save_path = os.path.join(os.path.abspath(os.getcwd()), "{}.wav".format(index_no))
			listener.recording(save_path, frames)
			index_no += 1
	except KeyboardInterrupt:
		print("KeyBoard Interrupt")
	except Exception as e:
		print(f"Keyboard Interrupt or {str(e)}")


def main(args):
	listener = Recorder(args)
	frames = []
	print("Recording...")
	try:
		while True:
			if listener.second == None:
				print("Recording without time limit... or press (ctrl+c) to interrupt recording")
				data = listener.stream.read(listener.chunk)
				frames.append(data)
			else:
				for i in range(0, int((listener.fs / listener.chunk) * listener.second)):
					data = listener.stream.read(listener.chunk)
					frames.append(data)
	except KeyboardInterrupt:
		print("KeyBoard Interrupt")
	except Exception as e:
		print(str(e))

	print("Record Finish...")
	listener.recording(args.save_path, frames)


if __name__=="__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--second", type=int, default=None, required=True)
	parser.add_argument("--save_path", type=str, default=None, required=False)
	parser.add_argument("--record", default=False, action='store_true', required=False)

	args = parser.parse_args()
	# if args.record:
	# 	# if args.save_path is None:
	# 		# raise Exception('need to set --save_path')
	record(args)
	# else:
		# if args.save_path is None:
			# raise Exception('need to set --save_path')
	main(args)
