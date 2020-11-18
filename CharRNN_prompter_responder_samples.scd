(
s.waitForBoot(
	{
		Ndef(\listener, {
			|fM=1.2, md=0.1, thresh=0.05|
			var onsets = Coyote.kr(in:SoundIn.ar(0), fastMul:1.2, minDur:0.1, thresh:0.05);
			var freqs = Tartini.kr(in: SoundIn.ar(0))[0];
			var clicks = MouseButton.kr(lag:0.0001);
			SendReply.kr(trig: Impulse.kr(6), cmdName: '/onset_freq', values: [onsets, freqs, clicks]);
		}).play;

		~addr = NetAddr.new("127.0.0.1", 2019);
		OSCdef.trace(false);
		OSCdef(\send_msg, { |msg|
			if(msg[3] == 1.0, {
				'seed pitch: '.postln; msg[4].asString.postln;
				"generating from RNN...".postln;
				~addr.sendMsg("/hello from SuperCollider", msg[4].cpsmidi.round(0.1).asString); // send RNN prompt to Python. change msg[4]... to eg '44.1' if using mousebutton trig
		})},'/onset_freq');

		// bank of samples to be triggered by RNN outputs:

		~a = Buffer.read(s, "path/to/wav");
		~b = Buffer.read(s, "path/to/wav");
		~c = Buffer.read(s, "path/to/wav");
		~d = Buffer.read(s, "path/to/wav");
		~e = Buffer.read(s, "path/to/wav");
		~f = Buffer.read(s, "path/to/wav");
		~g = Buffer.read(s, "path/to/wav");
		~h = Buffer.read(s, "path/to/wav");
		~i = Buffer.read(s, "path/to/wav");
		~j = Buffer.read(s, "path/to/wav");
		~k = Buffer.read(s, "path/to/wav");
		~l = Buffer.read(s, "path/to/wav");
		~m = Buffer.read(s, "path/to/wav");
		~n = Buffer.read(s, "path/to/wav");
		~o = Buffer.read(s, "path/to/wav");
		~p = Buffer.read(s, "path/to/wav");
		~q = Buffer.read(s, "path/to/wav");
		~r = Buffer.read(s, "path/to/wav");
		~s = Buffer.read(s, "path/to/wav");
		~t = Buffer.read(s, "path/to/wav");
		~u = Buffer.read(s, "path/to/wav");
		~v = Buffer.read(s, "path/to/wav");
		~w = Buffer.read(s, "path/to/wav");
		~x = Buffer.read(s, "path/to/wav");
		~y = Buffer.read(s, "path/to/wav");
		~z = Buffer.read(s, "path/to/wav");

		~aa = Buffer.read(s, "path/to/wav");
		~bb = Buffer.read(s, "path/to/wav");
		~cc = Buffer.read(s, "path/to/wav");
		~dd = Buffer.read(s, "path/to/wav");
		~ee = Buffer.read(s, "path/to/wav");
		~ff = Buffer.read(s, "path/to/wav");
		~gg = Buffer.read(s, "path/to/wav");
		~hh = Buffer.read(s, "path/to/wav");
		~ii = Buffer.read(s, "path/to/wav");
		~jj = Buffer.read(s, "path/to/wav");
		~kk = Buffer.read(s, "path/to/wav");
		~ll = Buffer.read(s, "path/to/wav");
		~mm = Buffer.read(s, "path/to/wav");
		~nn = Buffer.read(s, "path/to/wav");
		~oo = Buffer.read(s, "path/to/wav");
		~pp = Buffer.read(s, "path/to/wav");
		~qq = Buffer.read(s, "path/to/wav");
		~rr = Buffer.read(s, "path/to/wav");

		// map samples to midi values:

		~key = (44.0:~a, 45.0:~b, 46.0:~c, 47.0:~d, 48.0:~e, 49.0:~f, 50.0:~g, 51.0:~h, 52.0:~i, 53.0:~j, 54.0:~k, 55.0:~l, 56.0:~m, 57.0:~n, 58.0:~o, 59.0:~p, 60.0:~r, 61.0:~s, 62.0:~t, 63.0:~u, 64.0:~v, 65.0:~w, 66.0:~x, 67.0:~y, 68.0:~z, 69.0:~aa, 70.0:~bb, 71.0:~cc, 72.0:~dd, 73.0:~ee, 74.0:~ff, 75.0:~gg, 76.0:~hh, 77.0:~ii, 78.0:~jj, 79.0:~kk, 80.0:~ll, 81.0:~mm, 82.0:~nn, 83.0:~oo, 84.0:~pp, 85.0:~qq, 86.0:~rr);

		SynthDef(\samplePlayer, {
			|bufnum, rate=1.0, pointer, dur, mul|
			var sig = FreqShift.ar(WarpZ.ar(1, bufnum, pointer:Line.kr(start:0.1, end:0.9, dur:pointer, doneAction:2), mul:EnvGen.kr(Env.perc(0.05, dur), levelScale:mul)), freq:0);
			Out.ar([0, 1], sig);
		}).add;

		thisProcess.openUDPPort(1337);
		OSCdef.trace(false);
		OSCdef(\RNN_parser, {
			|msg|

			// separate incoming array from RNN into midinote (x) / dur (y) arrays:

			x=[]; ~pitch_amp=[]; ~note_on_off=[]; ~pitch=[]; ~amp=[]; ~note_on=[]; ~note_off=[];
			msg[1..].round(0.1).pairsDo({|a, b|x=x.add([a, b])});
			x.pairsDo({|c, d|~pitch_amp=~pitch_amp.add(c);~pitch_amp=~pitch_amp.flatten; ~note_on_off=~note_on_off.add(d); ~note_on_off=~note_on_off.flatten; });
			~pitch_amp.pairsDo({|e, f|~pitch=~pitch.add(e); ~amp=~amp.add(f);});
			~pitch=~pitch.round(1);
			~amp=~amp.dbamp.normalize.round(0.1);
			~amp=~amp.linlin(0.0, 1.0, 0.6, 0.8);
			// ~amp=~amp*1.5;
			~note_on_off.pairsDo({|g, h|~note_on=~note_on.add(g); ~note_off=~note_off.add(h);});
			~note_on=~note_on/95;
			~note_on=~note_on.round(0.1);
			~note_off.do({|el, index|if(el==10.0, {~note_off.put(index, 0.0)})});
			~note_off=~note_off/95;
			~note_off=~note_off.round(0.1);
			''.postln;
			'pitches:'.postln;
			~pitch.postln;
			''.postln;
			'amplitudes:'.postln;
			~amp.postln;
			''.postln;
			'note hold:'.postln;
			~note_on.postln;
			''.postln;
			'note off:'.postln;
			~note_off.postln;

			Pdef(\samps,
				Pbind(\instrument, \samplePlayer,
					\bufnum,
					Pseq([[~key.at(~pitch[1])], Rest(~note_off[1]), [~key.at(~pitch[2])], Rest(~note_off[2]), [~key.at(~pitch[3])], Rest(~note_off[3]), [~key.at(~pitch[4])], Rest(~note_off[4]), [~key.at(~pitch[5])], Rest(~note_off[5]), [~key.at(~pitch[6])], Rest(~note_off[6]), [~key.at(~pitch[7])], Rest(~note_off[7]), [~key.at(~pitch[8])], Rest(~note_off[8]), [~key.at(~pitch[9])], Rest(~note_off[9])], 1),
					\dur,
					Pseq([~note_on[1], Rest(~note_off[1]), ~note_on[2], Rest(~note_off[2]), ~note_on[3], Rest(~note_off[3]), ~note_on[4], Rest(~note_off[4]), ~note_on[5], Rest(~note_off[5]), ~note_on[6], Rest(~note_off[6]), ~note_on[7], Rest(~note_off[7]),~note_on[8], Rest(~note_off[8]),~note_on[9], Rest(~note_off[9])], 1),
					\mul,
					Pseq([~amp[1], Rest(~note_off[1]), ~amp[2], Rest(~note_off[2]), ~amp[3], Rest(~note_off[3]), ~amp[4], Rest(~note_off[4]), ~amp[5], Rest(~note_off[5]), ~amp[6], Rest(~note_off[6]), ~amp[7], Rest(~note_off[7]), ~amp[8], Rest(~note_off[8]), ~amp[9], Rest(~note_off[9])], 1),
					\pointer,
					Pseq([~note_on[1], Rest(~note_off[1]), ~note_on[2], Rest(~note_off[2]), ~note_on[3], Rest(~note_off[3]), ~note_on[4], Rest(~note_off[4]), ~note_on[5], Rest(~note_off[5]), ~note_on[6], Rest(~note_off[6]), ~note_on[7], Rest(~note_off[7]),~note_on[8], Rest(~note_off[8]),~note_on[9], Rest(~note_off[9])], 1)
			));
		},'/RNN1');

		Pdef(\samps).play;
	}
)
)