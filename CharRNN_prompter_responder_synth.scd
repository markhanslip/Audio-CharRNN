(
s.waitForBoot(
	{
		b = NetAddr.new("127.0.0.1", 2019);
		b.sendMsg("/hello from SuperCollider", '64.1, 60.1'); // kicks process off
		thisProcess.openUDPPort(1337);

		Ndef(\silence, {
			var detect = DetectSilence.ar(in: Ndef.ar(\stream, 2), amp:0.2, time:0.5, doneAction:0);
			SendReply.kr(Impulse.kr(2), cmdName: '/silence', values:detect);}).play;

		OSCdef.trace(false);
		OSCdef(\RNN_parser, {
			|msg|

			//var amp = Amplitude.kr(in:Ndef.ar(\stream, 2));
			x=[]; ~pitch_amp=[]; ~note_on_off=[]; ~pitch=[]; ~amp=[]; ~note_on=[]; ~note_off=[];
			msg[1..].round(0.1).pairsDo({|a, b|x=x.add([a, b])});
			x.pairsDo({|c, d|~pitch_amp=~pitch_amp.add(c);~pitch_amp=~pitch_amp.flatten; ~note_on_off=~note_on_off.add(d); ~note_on_off=~note_on_off.flatten; });
			~pitch_amp.pairsDo({|e, f|~pitch=~pitch.add(e); ~amp=~amp.add(f);});
			~pitch.do({|el, index|if(el==10.0, {~pitch.put(index, 44.0)})});

			//~pitch=~pitch.round(1).midicps;
			~amp=~amp.dbamp.normalize.round(0.1);
			~amp=~amp.linlin(0.0, 1.0, 0.4, 0.9);
			~note_on_off.pairsDo({|g, h|~note_on=~note_on.add(g); ~note_off=~note_off.add(h);});
			//~note_on=(~note_on/80);
			~note_off.do({|el, index|if(el==10.0, {~note_off.put(index, 0.0)})});
			//~note_off=~note_off/80;
			''.postln;
			'durations:'.postln;
			~note_on_offf=~note_on+~note_off;
			// ~note_on_offf = ~note_on_off.pairsDo({|i, j|~note_on_offf.add((i+j));});
			~note_on_offf.postln;
			''.postln;
			'pitches:'.postln;
			~pitch.postln;
			''.postln;
			'amplitudes:'.postln;
			~amp.postln;
			''.postln;
			'note hold:'.postln;
			~note_on=~note_on;
			~note_on.postln;
			''.postln;
			'note off:'.postln;
			~note_off=~note_off;
			~note_off.postln;
			// ~pitch=~pitch.round(1).midicps;

			Ndef(\stream, {
				|dDiv=85, freqMul=1|
				var freqs = Duty.kr(Dseq(~note_on_offf/dDiv, 1) reset:Dseq([~note_on_offf.sum/dDiv],  1), level:Dseq(~pitch.round(1).midicps, 1), doneAction:0);
				var durs = Duty.kr(Dseq(~note_on_offf/dDiv, 1) reset:Dseq([~note_on_offf.sum/dDiv],  1), level:Dseq(~note_on_offf, 1), doneAction:0);
				var amps = Duty.kr(Dseq(~note_on_offf/dDiv, 1) reset:Dseq([~note_on_offf.sum/dDiv],  1), level:Dseq(~amp, 1), doneAction:0);

				//var env = DemandEnvGen.kr(Dseq(~amp, 1), Dseq(~note_on_offf/dDiv, 1), doneAction:0);
				var env = EnvGen.kr(Env.new(levels:[0.01, amps, 0.0],times:[0.01, (~note_on_offf.sum)/dDiv]), doneAction:0);

				Limiter.ar(GVerb.ar(LPF.ar(in:LFSaw.ar(freqs*freqMul, mul:env), freq:SinOsc.kr(freq:freqs.cpsmidi.linlin(44.0, 84.0, 1.0, 8.0)).range(440,1760)), roomsize:6, revtime:3, damping:0.94, taillevel:1.0), level:0.95)}).play;

			Ndef(\stream).fadeTime=0.02;

			OSCdef(\modelPrompter, {
				|msg|
				if(msg[3] == 1, {
					b.sendMsg("/hello from SuperCollider", ~pitch.at(8).asString++', 60.1, '++~note_on.at(8).asString++', '++((~note_off.at(8)+10)).asString++', '++~pitch.at(9).asString++', 66.1');
				});
			}, '/silence');

		},'/RNN1');
	}
)
)