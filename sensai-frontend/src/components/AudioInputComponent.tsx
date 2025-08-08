"use client";

import { useState, useEffect, useRef } from 'react';
import { Mic, Play, Send, Pause, Trash2 } from 'lucide-react';

interface AudioInputComponentProps {
    onAudioSubmit: (audioBlob: Blob) => void;
    isSubmitting: boolean;
    maxDuration?: number;
    isDisabled?: boolean;
}

// Shared waveform rendering function to avoid duplication
const renderWaveformBar = (value: number, index: number, total: number, isPlayed: boolean = false) => {
    // Apply exponential scaling to emphasize differences
    const scaledHeight = Math.pow(value, 0.7) * 100;
    return (
        <div
            key={index}
            className="h-full flex items-end justify-center"
            style={{ width: `${100 / total}%` }}
        >
            <div
                className={`w-1 rounded-sm ${isPlayed ? 'bg-gradient-to-t from-white to-white/60' : 'bg-gradient-to-t from-white to-white/40'}`}
                style={{
                    height: `${Math.max(scaledHeight, 3)}%`
                }}
            ></div>
        </div>
    );
};

// Live Recording Waveform component
const LiveWaveform = ({ waveformData }: { waveformData: number[] }) => {
    return (
        <div className="w-full h-full flex items-end justify-between px-1 mb-4">
            {waveformData.map((value, index) =>
                renderWaveformBar(value, index, waveformData.length)
            )}
        </div>
    );
};

// Snapshot Waveform component for playback
const SnapshotWaveform = ({
    waveformData,
    playbackProgress
}: {
    waveformData: number[],
    playbackProgress: number
}) => {
    return (
        <div className="w-full h-full flex items-end justify-between relative px-1 mb-4">
            {/* Playback progress overlay */}
            <div
                className="absolute top-0 bottom-0 left-0 bg-white opacity-20 z-10 pointer-events-none"
                style={{ width: `${playbackProgress * 100}%` }}
            ></div>

            {waveformData.map((value, index) => {
                // Determine if this bar is in the played portion
                const isPlayed = (index / waveformData.length) < playbackProgress;
                return renderWaveformBar(value, index, waveformData.length, isPlayed);
            })}
        </div>
    );
};

// Function to get supported MIME type
const getSupportedMimeType = () => {
    const types = [
        'audio/webm',
        'audio/mp4',
        'audio/aac',
        'audio/ogg;codecs=opus',
        ''  // empty string means browser default
    ];

    for (const type of types) {
        if (!type || MediaRecorder.isTypeSupported(type)) {
            return type;
        }
    }
    return '';  // Return empty string as fallback (browser default)
};

export default function AudioInputComponent({
    onAudioSubmit,
    isSubmitting,
    maxDuration = 120,
    isDisabled = false
}: AudioInputComponentProps) {
    // Basic states
    const [isRecording, setIsRecording] = useState(false);
    const [recordingDuration, setRecordingDuration] = useState(0);
    const [audioBlob, setAudioBlob] = useState<Blob | null>(null);
    const [isPlaying, setIsPlaying] = useState(false);
    const [playbackProgress, setPlaybackProgress] = useState(0);
    const [showDeleteConfirmation, setShowDeleteConfirmation] = useState(false);

    // Separate waveform data states for live recording and snapshot
    const [liveWaveformData, setLiveWaveformData] = useState<number[]>([]);
    const [snapshotWaveformData, setSnapshotWaveformData] = useState<number[]>([]);

    // Live metrics state
    const [liveRms, setLiveRms] = useState(0);
    const [livePeak, setLivePeak] = useState(0);
    const [liveSpeakingRateWpm, setLiveSpeakingRateWpm] = useState<number | null>(null);
    const [liveNoiseLevel, setLiveNoiseLevel] = useState<'low' | 'moderate' | 'high'>('moderate');
    const [liveQualityHint, setLiveQualityHint] = useState('');

    // Live transcription state
    const [liveTranscript, setLiveTranscript] = useState('');
    const lastTranscriptSegmentRef = useRef('');
    const isPostingChunkRef = useRef(false);
    const lastChunkSentAtRef = useRef<number>(0);

    // Smoothing/throttling refs for metrics
    const smoothedRmsRef = useRef(0);
    const smoothedPeakRef = useRef(0);
    const smoothedWpmRef = useRef<number | null>(null);
    const lastMetricsUpdateRef = useRef<number>(0);
    const lastHintChangeAtRef = useRef<number>(0);

    // Refs
    const mediaRecorderRef = useRef<MediaRecorder | null>(null);
    const audioChunksRef = useRef<Blob[]>([]);
    const audioContextRef = useRef<AudioContext | null>(null);
    const analyserRef = useRef<AnalyserNode | null>(null);
    const audioPlayerRef = useRef<HTMLAudioElement | null>(null);
    const animationFrameRef = useRef<number | null>(null);
    const timerRef = useRef<NodeJS.Timeout | null>(null);
    const streamRef = useRef<MediaStream | null>(null);

    // Initialize audio player
    useEffect(() => {
        return () => {
            // Clean up on unmount
            if (streamRef.current) {
                streamRef.current.getTracks().forEach(track => track.stop());
            }

            if (animationFrameRef.current) {
                cancelAnimationFrame(animationFrameRef.current);
            }

            if (timerRef.current) {
                clearInterval(timerRef.current);
            }
        };
    }, []);

    // Start recording function
    const startRecording = async () => {
        try {
            // Reset everything
            setLiveWaveformData([]);
            setSnapshotWaveformData([]);
            setAudioBlob(null);
            audioChunksRef.current = [];

            // Create audio context
            const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
            audioContextRef.current = audioContext;

            // Get microphone stream
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            streamRef.current = stream;

            // Create and configure analyser
            const analyser = audioContext.createAnalyser();
            analyser.fftSize = 256;
            analyserRef.current = analyser;

            // Connect microphone stream to analyser
            const source = audioContext.createMediaStreamSource(stream);
            source.connect(analyser);

            // Replace the MediaRecorder initialization with:
            const mimeType = getSupportedMimeType();
            const mediaRecorder = new MediaRecorder(stream,
                mimeType ? { mimeType } : undefined
            );
            mediaRecorderRef.current = mediaRecorder;

            // When data becomes available, add it to our array and transcribe chunk
            mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    audioChunksRef.current.push(event.data);
                    if (isRecording) {
                        transcribeChunk(event.data);
                    }
                }
            };

            // When recording stops
            mediaRecorder.onstop = () => {
                // Create audio blob from recorded chunks
                const audioBlob = new Blob(audioChunksRef.current, { type: mimeType });
                setAudioBlob(audioBlob);

                // Set up audio player
                if (audioPlayerRef.current) {
                    const audioUrl = URL.createObjectURL(audioBlob);
                    audioPlayerRef.current.src = audioUrl;
                } else {
                    const audioPlayer = new Audio();
                    audioPlayer.src = URL.createObjectURL(audioBlob);
                    audioPlayerRef.current = audioPlayer;

                    // Set up event listeners
                    audioPlayer.addEventListener('ended', () => {
                        setIsPlaying(false);
                        setPlaybackProgress(0);
                    });

                    audioPlayer.addEventListener('timeupdate', () => {
                        if (audioPlayer.duration) {
                            setPlaybackProgress(audioPlayer.currentTime / audioPlayer.duration);
                        }
                    });
                }

                // Generate snapshot waveform from the recorded audio
                generateWaveformFromAudio(audioBlob);

                // Clean up
                if (streamRef.current) {
                    streamRef.current.getTracks().forEach(track => track.stop());
                }
            };

            // Set recording state first
            setIsRecording(true);
            setLiveTranscript("");

            // Start recording with 1s timeslices for live transcription
            mediaRecorder.start(1000);
            setRecordingDuration(0);

            // Set timer for recording duration
            timerRef.current = setInterval(() => {
                setRecordingDuration(prev => {
                    if (prev >= maxDuration - 1) {
                        stopRecording();
                        return maxDuration;
                    }
                    return prev + 1;
                });
            }, 1000);

            // Start visualization after setting recording state
            updateLiveWaveform(analyser);
        } catch (error) {
            console.error('Error starting recording:', error);
        }
    };

    // Update the live waveform during recording
    const updateLiveWaveform = (analyser: AnalyserNode) => {
        // This function gets called continuously by requestAnimationFrame
        const draw = () => {
            // Get time domain data for waveform visualization
            const bufferLength = analyser.fftSize;
            const dataArray = new Uint8Array(bufferLength);
            analyser.getByteTimeDomainData(dataArray);

            // Process the data to create the waveform (sample to ~40 points for visualization)
            const newWaveformData = [];
            const step = Math.floor(bufferLength / 40) || 1;

            for (let i = 0; i < bufferLength; i += step) {
                let sum = 0;
                let count = 0;

                // Average a few points together
                for (let j = 0; j < step && i + j < bufferLength; j++) {
                    // For time domain data, we want the absolute deviation from 128 (midpoint)
                    sum += Math.abs(dataArray[i + j] - 128);
                    count++;
                }

                // Normalize to 0-1 range
                const average = count > 0 ? sum / count / 128 : 0;
                newWaveformData.push(average);

                // Limit to 40 data points
                if (newWaveformData.length >= 40) break;
            }

            // Update live waveform state
            setLiveWaveformData(newWaveformData);

            // Compute live metrics from time-domain data
            // Convert 0..255 to centered [-1, 1]
            let sumSquares = 0;
            let peakAbs = 0;
            let activeSamples = 0;
            for (let i = 0; i < bufferLength; i++) {
                const centered = (dataArray[i] - 128) / 128;
                const absVal = Math.abs(centered);
                sumSquares += centered * centered;
                if (absVal > peakAbs) peakAbs = absVal;
                if (absVal > 0.06) activeSamples += 1; // simple VAD threshold
            }

            const rms = Math.sqrt(sumSquares / bufferLength);
            // Rough speaking rate estimate mapped from activity ratio
            const activeRatio = activeSamples / bufferLength; // 0..1
            // Map activity ratio to WPM range ~90..190 (heuristic)
            const estimatedWpm = Math.round(90 + (activeRatio * 100));

            // Exponential moving average smoothing
            const alpha = 0.1; // smoothing factor (less sensitive)
            smoothedRmsRef.current = alpha * rms + (1 - alpha) * smoothedRmsRef.current;
            smoothedPeakRef.current = alpha * peakAbs + (1 - alpha) * smoothedPeakRef.current;
            smoothedWpmRef.current = Number.isFinite(estimatedWpm)
                ? (smoothedWpmRef.current == null
                    ? estimatedWpm
                    : Math.round(alpha * estimatedWpm + (1 - alpha) * smoothedWpmRef.current))
                : smoothedWpmRef.current;

            // Throttle state updates to ~3.3Hz
            const now = Date.now();
            if (now - lastMetricsUpdateRef.current > 300) {
                lastMetricsUpdateRef.current = now;
                setLiveRms(smoothedRmsRef.current);
                setLivePeak(smoothedPeakRef.current);
                setLiveSpeakingRateWpm(smoothedWpmRef.current);

                // Noise level heuristic based on smoothed metrics
                let noise: 'low' | 'moderate' | 'high' = 'moderate';
                if (smoothedRmsRef.current < 0.015 && smoothedPeakRef.current < 0.06) noise = 'low';
                else if (smoothedRmsRef.current > 0.12 && smoothedPeakRef.current > 0.3) noise = 'high';
                setLiveNoiseLevel(noise);

                // Minimal, actionable hint with debounce
                let hint = '';
                if (smoothedPeakRef.current < 0.05) hint = 'Speak closer to the mic';
                else if (smoothedPeakRef.current > 0.9) hint = 'Reduce input volume';
                else if (noise === 'high') hint = 'Reduce background noise';
                else if (
                    smoothedWpmRef.current && (smoothedWpmRef.current < 100 || smoothedWpmRef.current > 200)
                ) hint = 'Adjust speaking pace';

                if (hint !== liveQualityHint && now - lastHintChangeAtRef.current > 1800) {
                    lastHintChangeAtRef.current = now;
                    setLiveQualityHint(hint);
                }
            }

            // Continue the animation loop
            animationFrameRef.current = requestAnimationFrame(draw);
        };

        // Start the animation loop
        animationFrameRef.current = requestAnimationFrame(draw);
    };

    // Stop recording
    const stopRecording = () => {
        if (mediaRecorderRef.current && isRecording) {
            mediaRecorderRef.current.stop();
            setIsRecording(false);

            if (timerRef.current) {
                clearInterval(timerRef.current);
            }

            // Cancel animation frame here
            if (animationFrameRef.current) {
                cancelAnimationFrame(animationFrameRef.current);
            }
        }
    };

    // Convert AudioBuffer to WAV (16-bit PCM)
    const convertAudioBufferToWav = (audioBuffer: AudioBuffer) => {
        const numOfChan = audioBuffer.numberOfChannels;
        const length = audioBuffer.length * numOfChan * 2;
        const buffer = new ArrayBuffer(44 + length);
        const view = new DataView(buffer);
        const sampleRate = audioBuffer.sampleRate;
        const channels: Float32Array[] = [];
        for (let i = 0; i < numOfChan; i++) channels.push(audioBuffer.getChannelData(i));
        writeString(view, 0, 'RIFF');
        view.setUint32(4, 36 + length, true);
        writeString(view, 8, 'WAVE');
        writeString(view, 12, 'fmt ');
        view.setUint32(16, 16, true);
        view.setUint16(20, 1, true);
        view.setUint16(22, numOfChan, true);
        view.setUint32(24, sampleRate, true);
        view.setUint32(28, sampleRate * numOfChan * 2, true);
        view.setUint16(32, numOfChan * 2, true);
        view.setUint16(34, 16, true);
        writeString(view, 36, 'data');
        view.setUint32(40, length, true);
        let pos = 44;
        for (let i = 0; i < audioBuffer.length; i++) {
            for (let ch = 0; ch < numOfChan; ch++) {
                const sample = Math.max(-1, Math.min(1, channels[ch][i]));
                const value = sample < 0 ? sample * 0x8000 : sample * 0x7FFF;
                view.setInt16(pos, value, true);
                pos += 2;
            }
        }
        return buffer;
    };

    const writeString = (view: DataView, offset: number, str: string) => {
        for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
    };

    // Send chunk for live transcription
    const transcribeChunk = async (blob: Blob) => {
        const now = Date.now();
        // Throttle to once every 1500ms
        if (now - lastChunkSentAtRef.current < 1500 || isPostingChunkRef.current) return;
        isPostingChunkRef.current = true;
        try {
            const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
            const arrayBuffer = await blob.arrayBuffer();
            const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
            const wavBuffer = convertAudioBufferToWav(audioBuffer);
            const wavBlob = new Blob([wavBuffer], { type: 'audio/wav' });
            const base64Data = await new Promise<string>((resolve, reject) => {
                const reader = new FileReader();
                reader.onloadend = () => {
                    const result = reader.result as string;
                    resolve(result.split(',')[1]);
                };
                reader.onerror = reject;
                reader.readAsDataURL(wavBlob);
            });
            const resp = await fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL}/ai/transcribe/chunk`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ audio_base64: base64Data })
            });
            if (resp.ok) {
                const data = await resp.json();
                const text: string = (data?.text || '').trim();
                if (text && text !== lastTranscriptSegmentRef.current) {
                    lastTranscriptSegmentRef.current = text;
                    setLiveTranscript(prev => (prev ? `${prev} ${text}` : text));
                }
                lastChunkSentAtRef.current = now;
            }
        } catch (err) {
            // Silently ignore to avoid UI noise during recording
        } finally {
            isPostingChunkRef.current = false;
        }
    };

    // Toggle audio playback
    const togglePlayback = () => {
        if (!audioPlayerRef.current || !audioBlob) return;

        if (isPlaying) {
            audioPlayerRef.current.pause();
            setIsPlaying(false);
        } else {
            audioPlayerRef.current.play();
            setIsPlaying(true);

            // If snapshot waveform data is empty, try to generate it from the recorded audio
            if (snapshotWaveformData.length === 0 && audioBlob) {
                generateWaveformFromAudio(audioBlob);
            }
        }
    };

    // Function to generate snapshot waveform data from an audio blob
    const generateWaveformFromAudio = async (blob: Blob) => {
        try {
            // Create a new audio context
            const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();

            // Convert blob to array buffer
            const arrayBuffer = await blob.arrayBuffer();

            // Decode the audio data
            const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);

            // Get the channel data
            const channelData = audioBuffer.getChannelData(0);

            // Sample the audio data to create waveform
            const samples = 40;
            const blockSize = Math.floor(channelData.length / samples);
            const sampledData = [];

            for (let i = 0; i < samples; i++) {
                let sum = 0;
                for (let j = 0; j < blockSize; j++) {
                    const index = (i * blockSize) + j;
                    if (index < channelData.length) {
                        sum += Math.abs(channelData[index]);
                    }
                }
                // Average and normalize (audio data is -1 to 1)
                // Use a different normalization factor to accentuate differences
                const normalized = sum / (blockSize * 0.8); // Increase visibility by reducing divisor
                sampledData.push(Math.min(normalized, 1)); // Cap at 1
            }

            // Apply some smoothing to make the waveform look more natural
            const smoothedData = [];
            for (let i = 0; i < sampledData.length; i++) {
                const prev = i > 0 ? sampledData[i - 1] : sampledData[i];
                const current = sampledData[i];
                const next = i < sampledData.length - 1 ? sampledData[i + 1] : sampledData[i];
                // Weighted average with current sample having more weight
                smoothedData.push((prev * 0.2) + (current * 0.6) + (next * 0.2));
            }

            // Update snapshot waveform data
            setSnapshotWaveformData(smoothedData);

            // Close the audio context
            audioContext.close();
        } catch (error) {
            console.error('Error generating waveform:', error);
        }
    };

    // Seek in audio playback
    const handleSeek = (e: React.MouseEvent<HTMLDivElement>) => {
        if (!audioPlayerRef.current || !audioBlob) return;

        const container = e.currentTarget;
        const rect = container.getBoundingClientRect();
        const clickPosition = e.clientX - rect.left;
        const containerWidth = rect.width;
        const seekPercentage = clickPosition / containerWidth;

        if (audioPlayerRef.current) {
            audioPlayerRef.current.currentTime = seekPercentage * audioPlayerRef.current.duration;
            setPlaybackProgress(seekPercentage);

            if (isPlaying) {
                audioPlayerRef.current.play();
            }
        }
    };

    // Submit recorded audio
    const handleSubmit = () => {
        if (audioBlob && !isSubmitting) {
            onAudioSubmit(audioBlob);
            setAudioBlob(null);
            setLiveWaveformData([]);
            setSnapshotWaveformData([]);
            // Close the delete confirmation dialog if it's open
            setShowDeleteConfirmation(false);
        }
    };

    // Format time for display
    const formatTime = (seconds: number) => {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}:${secs < 10 ? '0' : ''}${secs}`;
    };

    // New function to handle delete button click
    const handleDeleteClick = () => {
        setShowDeleteConfirmation(true);
    };

    // New function to confirm deletion
    const confirmDelete = () => {
        // Stop playback if it's playing
        if (isPlaying && audioPlayerRef.current) {
            audioPlayerRef.current.pause();
            setIsPlaying(false);
        }

        // Reset all audio-related states
        setAudioBlob(null);
        setLiveWaveformData([]);
        setSnapshotWaveformData([]);
        setPlaybackProgress(0);

        // Close confirmation dialog
        setShowDeleteConfirmation(false);

        // Clear audio player source if it exists
        if (audioPlayerRef.current) {
            audioPlayerRef.current.src = '';
        }
    };

    // New function to cancel deletion
    const cancelDelete = () => {
        setShowDeleteConfirmation(false);
    };

    return (
        <div className="relative">
            {/* Recording status and timer (moved up space for transcript) */}
            {isRecording && (
                <div className="absolute -top-12 left-0 right-0 text-center flex items-center justify-center z-20">
                    <div className="bg-black/80 rounded-full px-4 py-2 shadow-md flex items-center">
                        <div className="w-2 h-2 bg-red-500 rounded-full mr-2 animate-pulse"></div>
                        <span className="text-red-500 font-light text-sm">Recording {formatTime(recordingDuration)}</span>
                    </div>
                </div>
            )}

            {/* Delete confirmation dialog */}
            {showDeleteConfirmation && (
                <div className="absolute -top-20 left-0 right-0 bg-[#222222] rounded-lg p-3 shadow-lg z-20">
                    <p className="text-white text-sm mb-2">Are you sure you want to delete this recording?</p>
                    <div className="flex justify-end space-x-2">
                        <button
                            className="text-white text-xs bg-transparent hover:bg-[#333333] px-2 py-1 rounded-md cursor-pointer"
                            onClick={cancelDelete}
                        >
                            Cancel
                        </button>
                        <button
                            className="text-white text-xs bg-red-500 hover:bg-red-600 px-2 py-1 rounded-md cursor-pointer"
                            onClick={confirmDelete}
                        >
                            Delete
                        </button>
                    </div>
                </div>
            )}

            {/* Live transcript and quality hint (minimal, above bar) */}
            {isRecording && (
                <div className="mb-2 px-1">
                    {liveTranscript && (
                        <div className="text-white/80 text-xs font-light truncate" title={liveTranscript}>
                            {liveTranscript}
                        </div>
                    )}
                    {liveQualityHint && (
                        <div className="text-white/60 text-[11px] font-light mt-1">
                            {liveQualityHint}
                        </div>
                    )}
                </div>
            )}

            {/* Main container */}
            <div className="relative flex items-center bg-[#111111] rounded-full overflow-hidden border border-[#222222] px-3 py-2">
                {/* Record/Play/Stop button */}
                {isSubmitting ? (
                    <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                ) : (
                    <button
                        className="w-8 h-8 sm:w-10 sm:h-10 flex-shrink-0 rounded-full flex items-center justify-center bg-[#222222] text-white hover:bg-[#333333] cursor-pointer mr-3"
                        onClick={isRecording ? stopRecording : audioBlob ? togglePlayback : startRecording}
                        disabled={isDisabled}
                        type="button"
                    >
                        {isRecording ? (
                            <div className="w-3 h-3 bg-white"></div>
                        ) : audioBlob ? (
                            isPlaying ? <Pause size={16} /> : <><Play size={14} className="sm:hidden" /> <Play size={16} className="hidden sm:block" /></>
                        ) : (
                            <Mic size={16} />
                        )}
                    </button>
                )}

                {/* Redesigned layout with waveform extending full width */}
                <div className="flex-1 relative">
                    {/* Flex container for waveform and submit button */}
                    <div className="flex w-full items-center">
                        {/* Waveform container that adjusts width based on recording state */}
                        <div
                            className={`h-10 flex items-center justify-center relative cursor-pointer ${audioBlob
                                ? 'flex-1 max-w-[calc(100%-80px)] sm:max-w-none' // Add max-width constraint on mobile
                                : 'w-full'
                                }`}
                            onClick={audioBlob && !isRecording ? handleSeek : undefined}
                        >
                            {/* Waveform visualization - show different components based on state */}
                            {isRecording && liveWaveformData.length > 0 ? (
                                <LiveWaveform waveformData={liveWaveformData} />
                            ) : audioBlob && snapshotWaveformData.length > 0 ? (
                                <SnapshotWaveform
                                    waveformData={snapshotWaveformData}
                                    playbackProgress={playbackProgress}
                                />
                            ) : (
                                <div className="text-gray-400 text-xs sm:text-sm">Click the microphone to start recording</div>
                            )}
                        </div>

                        {/* Action buttons - added delete button */}
                        {audioBlob && (
                            <div className="ml-2 sm:ml-3 flex-shrink-0 flex space-x-1 sm:space-x-2">
                                {/* Delete button */}
                                <button
                                    className="w-8 h-8 sm:w-10 sm:h-10 rounded-full flex items-center justify-center bg-[#222222] text-white hover:bg-[#333333] cursor-pointer"
                                    onClick={handleDeleteClick}
                                    disabled={isSubmitting || isDisabled}
                                    aria-label="Delete audio"
                                    type="button"
                                >
                                    <Trash2 size={14} className="sm:hidden" />
                                    <Trash2 size={16} className="hidden sm:block" />
                                </button>

                                {/* Submit button */}
                                <button
                                    className="w-8 h-8 sm:w-10 sm:h-10 rounded-full flex items-center justify-center bg-white cursor-pointer"
                                    onClick={handleSubmit}
                                    disabled={isSubmitting || isDisabled}
                                    aria-label="Submit audio"
                                    type="button"
                                >
                                    {isSubmitting ? (
                                        <div className="w-4 h-4 sm:w-5 sm:h-5 border-2 border-black border-t-transparent rounded-full animate-spin"></div>
                                    ) : (
                                        <>
                                            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" className="sm:hidden">
                                                <path d="M5 12H19M19 12L12 5M19 12L12 19" stroke="black" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                                            </svg>
                                            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" className="hidden sm:block">
                                                <path d="M5 12H19M19 12L12 5M19 12L12 19" stroke="black" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                                            </svg>
                                        </>
                                    )}
                                </button>
                            </div>
                        )}
                    </div>
                </div>
            </div>

            {/* (moved transcript above) */}
        </div>
    );
} 