/**
 * VIGIL Audio Alarm Engine
 * Uses Web Audio API to generate alarm sounds client-side.
 */

type SoundType = 'alert' | 'siren' | 'beep'

let audioCtx: AudioContext | null = null

function getContext(): AudioContext {
  if (!audioCtx || audioCtx.state === 'closed') {
    audioCtx = new AudioContext()
  }
  if (audioCtx.state === 'suspended') {
    audioCtx.resume()
  }
  return audioCtx
}

/** Generate an alert tone — two rapid descending bursts */
function playAlert(ctx: AudioContext, volume: number) {
  const now = ctx.currentTime
  const gain = ctx.createGain()
  gain.connect(ctx.destination)
  gain.gain.setValueAtTime(volume * 0.4, now)

  // Burst 1
  const o1 = ctx.createOscillator()
  o1.type = 'square'
  o1.frequency.setValueAtTime(880, now)
  o1.frequency.linearRampToValueAtTime(440, now + 0.15)
  o1.connect(gain)
  o1.start(now)
  o1.stop(now + 0.15)

  // Burst 2
  const o2 = ctx.createOscillator()
  o2.type = 'square'
  o2.frequency.setValueAtTime(880, now + 0.2)
  o2.frequency.linearRampToValueAtTime(440, now + 0.35)
  o2.connect(gain)
  o2.start(now + 0.2)
  o2.stop(now + 0.35)

  gain.gain.setValueAtTime(0, now + 0.4)
}

/** Generate a siren — smooth up-down sweep */
function playSiren(ctx: AudioContext, volume: number) {
  const now = ctx.currentTime
  const gain = ctx.createGain()
  gain.connect(ctx.destination)
  gain.gain.setValueAtTime(volume * 0.35, now)
  gain.gain.setValueAtTime(0, now + 1.0)

  const osc = ctx.createOscillator()
  osc.type = 'sawtooth'
  osc.frequency.setValueAtTime(400, now)
  osc.frequency.linearRampToValueAtTime(900, now + 0.5)
  osc.frequency.linearRampToValueAtTime(400, now + 1.0)
  osc.connect(gain)
  osc.start(now)
  osc.stop(now + 1.0)
}

/** Generate a simple beep */
function playBeep(ctx: AudioContext, volume: number) {
  const now = ctx.currentTime
  const gain = ctx.createGain()
  gain.connect(ctx.destination)
  gain.gain.setValueAtTime(volume * 0.3, now)
  gain.gain.exponentialRampToValueAtTime(0.001, now + 0.3)

  const osc = ctx.createOscillator()
  osc.type = 'sine'
  osc.frequency.setValueAtTime(800, now)
  osc.connect(gain)
  osc.start(now)
  osc.stop(now + 0.3)
}

/** Play the alarm sound locally using Web Audio API */
export function playAlarm(soundType: SoundType = 'alert', volume = 0.7): void {
  try {
    const ctx = getContext()
    switch (soundType) {
      case 'siren':
        playSiren(ctx, volume)
        break
      case 'beep':
        playBeep(ctx, volume)
        break
      case 'alert':
      default:
        playAlert(ctx, volume)
        break
    }
  } catch (e) {
    console.warn('Audio playback failed:', e)
  }
}

/** Play a quick success chirp */
export function playSuccess(): void {
  try {
    const ctx = getContext()
    const now = ctx.currentTime
    const gain = ctx.createGain()
    gain.connect(ctx.destination)
    gain.gain.setValueAtTime(0.1, now)
    gain.gain.exponentialRampToValueAtTime(0.001, now + 0.15)

    const osc = ctx.createOscillator()
    osc.type = 'sine'
    osc.frequency.setValueAtTime(600, now)
    osc.frequency.linearRampToValueAtTime(1200, now + 0.1)
    osc.connect(gain)
    osc.start(now)
    osc.stop(now + 0.15)
  } catch (e) {
    console.warn('Audio playback failed:', e)
  }
}
