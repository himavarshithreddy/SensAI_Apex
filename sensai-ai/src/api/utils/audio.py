import base64
import io
import logging
import openai
import tempfile
import os
import wave
import ssl
import urllib.request
import statistics
import math
from typing import List, Dict, Tuple, Optional
from api.settings import settings

logger = logging.getLogger(__name__)

# Global variable to cache the Whisper model
_whisper_model = None


def prepare_audio_input_for_ai(audio_data: bytes):
    return base64.b64encode(audio_data).decode("utf-8")


def analyze_audio_features(audio_path: str) -> dict:
    """
    Analyze audio features for delivery and clarity assessment
    
    Args:
        audio_path: Path to the audio file
        
    Returns:
        Dictionary with audio features for delivery analysis
    """
    try:
        import numpy as np
        
        analysis = {
            "speaking_rate": "unknown",
            "pause_pattern": "unknown", 
            "volume_consistency": "unknown",
            "energy_level": "unknown",
            "pace_assessment": "unknown"
        }
        
        # Analyze basic audio properties
        with wave.open(audio_path, 'rb') as wav_file:
            frames = wav_file.readframes(-1)
            sample_rate = wav_file.getframerate()
            duration = len(frames) / sample_rate / wav_file.getsampwidth() / wav_file.getnchannels()
            
            # Convert to numpy array for analysis
            audio_data = np.frombuffer(frames, dtype=np.int16)
            
            # Calculate speaking rate (very basic - words per minute estimation)
            # This is rough - would need more sophisticated analysis for accuracy
            if duration > 0:
                # Estimate based on audio activity vs silence
                energy = np.abs(audio_data)
                threshold = np.percentile(energy, 20)  # 20th percentile as silence threshold
                active_samples = np.sum(energy > threshold)
                active_ratio = active_samples / len(audio_data)
                
                # Rough speaking rate estimation
                estimated_words_per_second = active_ratio * 2.5  # Average speaking rate
                speaking_rate_wpm = estimated_words_per_second * 60
                
                if speaking_rate_wpm < 120:
                    analysis["speaking_rate"] = "slow"
                    analysis["pace_assessment"] = "Consider speaking slightly faster for better engagement"
                elif speaking_rate_wpm > 180:
                    analysis["speaking_rate"] = "fast"  
                    analysis["pace_assessment"] = "Consider slowing down for better clarity"
                else:
                    analysis["speaking_rate"] = "normal"
                    analysis["pace_assessment"] = "Good speaking pace"
                
                # Volume consistency analysis
                volume_std = np.std(energy)
                volume_mean = np.mean(energy)
                if volume_mean > 0:
                    volume_cv = volume_std / volume_mean
                    if volume_cv < 0.5:
                        analysis["volume_consistency"] = "consistent"
                    elif volume_cv < 1.0:
                        analysis["volume_consistency"] = "somewhat_variable"
                    else:
                        analysis["volume_consistency"] = "highly_variable"
                
                # Energy level assessment
                if volume_mean < np.percentile(energy, 30):
                    analysis["energy_level"] = "low - consider speaking with more confidence"
                elif volume_mean > np.percentile(energy, 70):
                    analysis["energy_level"] = "high - good energy and confidence"
                else:
                    analysis["energy_level"] = "moderate"
            
        logger.info(f"Audio feature analysis completed: {analysis}")
        return analysis
        
    except Exception as e:
        logger.warning(f"Error analyzing audio features: {e}")
        return {
            "speaking_rate": "unknown",
            "pause_pattern": "unknown",
            "volume_consistency": "unknown", 
            "energy_level": "unknown",
            "pace_assessment": "Audio analysis unavailable"
        }


def analyze_whisper_metrics(whisper_result) -> dict:
    """
    Enhanced Whisper metrics analysis with advanced statistical modeling
    
    Args:
        whisper_result: Complete Whisper transcription result with segments and metadata
        
    Returns:
        Dictionary with comprehensive metrics and statistical analysis
    """
    try:
        analysis = {
            # Core confidence metrics
            "overall_confidence": "unknown",
            "confidence_score": 0.0,
            "confidence_stability": "unknown",
            "confidence_trend": "unknown",
            
            # Fluency and delivery
            "speaking_fluency": "unknown",
            "fluency_score": 0.0,
            "delivery_consistency": "unknown",
            "energy_variation": "unknown",
            
            # Pace and timing
            "pace_analysis": "unknown",
            "pace_score": 0.0,
            "speech_rhythm": "unknown",
            "pause_patterns": "unknown",
            
            # Content quality
            "pronunciation_clarity": "unknown",
            "clarity_score": 0.0,
            "filler_analysis": "unknown",
            "vocabulary_confidence": "unknown",
            
            # Statistical measures
            "confidence_percentiles": "unknown",
            "performance_indicators": "unknown",
            "improvement_metrics": "unknown",
            "comparative_analysis": "unknown"
        }
        
        segments = whisper_result.get("segments", [])
        if not segments:
            return analysis
            
        # Enhanced data collection with statistical tracking
        confidence_data = []
        word_timing_data = []
        pause_data = []
        segment_quality = []
        filler_positions = []
        
        # Advanced filler detection with context
        filler_words = {
            "basic": ["um", "uh", "er", "ah", "hmm"],
            "hedge": ["like", "you know", "i mean", "kind of", "sort of"],
            "transition": ["basically", "actually", "so", "well", "now"],
            "emphasis": ["really", "very", "totally", "definitely"]
        }
        
        all_fillers = [word for category in filler_words.values() for word in category]
        filler_counts = {category: 0 for category in filler_words}
        total_words = 0
        total_speech_time = 0
        
        # Confidence tracking with temporal analysis
        confidence_timeline = []
        prev_end_time = 0
        
        for segment_idx, segment in enumerate(segments):
            # Enhanced segment-level analysis
            avg_logprob = segment.get("avg_logprob", -1.0)
            compression_ratio = segment.get("compression_ratio", 1.0)
            no_speech_prob = segment.get("no_speech_prob", 1.0)
            segment_start = segment.get("start", 0)
            segment_end = segment.get("end", 0)
            segment_duration = segment_end - segment_start
            total_speech_time += segment_duration
            
            # Advanced confidence calculation with normalization
            # Whisper logprob typically ranges from -3.0 to 0.0
            raw_confidence = max(0, min(1, (avg_logprob + 3.0) / 3.0))
            
            # Quality adjustment based on compression ratio and no-speech probability
            quality_factor = 1.0
            if compression_ratio > 2.4:  # High compression indicates possible hallucination
                quality_factor *= 0.8
            if no_speech_prob > 0.6:  # High no-speech probability
                quality_factor *= 0.7
                
            adjusted_confidence = raw_confidence * quality_factor
            confidence_data.append({
                "timestamp": segment_start,
                "confidence": adjusted_confidence,
                "raw_confidence": raw_confidence,
                "quality_factor": quality_factor,
                "segment_idx": segment_idx
            })
            
            segment_quality.append({
                "confidence": adjusted_confidence,
                "compression_ratio": compression_ratio,
                "no_speech_prob": no_speech_prob,
                "duration": segment_duration
            })
            
            # Word-level enhanced analysis
            words = segment.get("words", [])
            for word_idx, word_info in enumerate(words):
                word_text = word_info.get("word", "").strip().lower()
                start_time = word_info.get("start", 0)
                end_time = word_info.get("end", 0)
                word_prob = word_info.get("probability", avg_logprob)
                
                total_words += 1
                
                # Enhanced word timing analysis
                word_duration = end_time - start_time
                if word_duration > 0:
                    word_timing_data.append({
                        "word": word_text,
                        "duration": word_duration,
                        "start": start_time,
                        "end": end_time,
                        "confidence": word_prob,
                        "segment_idx": segment_idx,
                        "word_idx": word_idx
                    })
                
                # Advanced pause analysis with context
                if prev_end_time > 0:
                    pause_duration = start_time - prev_end_time
                    if pause_duration > 0.05:  # 50ms minimum pause
                        pause_data.append({
                            "duration": pause_duration,
                            "position": start_time,
                            "context": "word_boundary",
                            "significance": "high" if pause_duration > 0.5 else "normal" if pause_duration > 0.2 else "minimal"
                        })
                
                prev_end_time = end_time
                
                # Enhanced filler detection with categorization
                for category, fillers in filler_words.items():
                    if any(filler in word_text for filler in fillers):
                        filler_counts[category] += 1
                        filler_positions.append({
                            "word": word_text,
                            "category": category,
                            "timestamp": start_time,
                            "confidence": word_prob
                        })
                        break
        
        # Extract all words data for pronunciation analysis
        words_data = []
        for segment in segments:
            words = segment.get("words", [])
            for word_info in words:
                words_data.append({
                    "word": word_info.get("word", "").strip(),
                    "probability": word_info.get("probability", 0),
                    "start": word_info.get("start", 0),
                    "end": word_info.get("end", 0)
                })
        
        # ENHANCED STATISTICAL ANALYSIS
        if confidence_data:
            confidences = [item["confidence"] for item in confidence_data]
            
            # Advanced confidence metrics
            avg_confidence = sum(confidences) / len(confidences)
            confidence_std = (sum((c - avg_confidence)**2 for c in confidences) / len(confidences))**0.5
            confidence_min = min(confidences)
            confidence_max = max(confidences)
            
            # Percentile analysis for detailed assessment
            sorted_conf = sorted(confidences)
            n = len(sorted_conf)
            p25 = sorted_conf[int(0.25 * n)]
            p50 = sorted_conf[int(0.50 * n)]  # median
            p75 = sorted_conf[int(0.75 * n)]
            p90 = sorted_conf[int(0.90 * n)]
            
            analysis["confidence_score"] = avg_confidence
            analysis["confidence_percentiles"] = f"P25:{p25:.2f} P50:{p50:.2f} P75:{p75:.2f} P90:{p90:.2f}"
            
            # Confidence trend analysis (improvement/decline over time)
            if len(confidence_data) >= 3:
                first_third = confidences[:len(confidences)//3]
                last_third = confidences[-len(confidences)//3:]
                trend_change = (sum(last_third)/len(last_third)) - (sum(first_third)/len(first_third))
                
                if trend_change > 0.1:
                    analysis["confidence_trend"] = f"improving (+{trend_change:.2f})"
                elif trend_change < -0.1:
                    analysis["confidence_trend"] = f"declining ({trend_change:.2f})"
                else:
                    analysis["confidence_trend"] = "stable"
            
            # Overall confidence with statistical context
            if avg_confidence >= 0.85:
                analysis["overall_confidence"] = f"excellent (top 10% performance, σ={confidence_std:.2f})"
            elif avg_confidence >= 0.75:
                analysis["overall_confidence"] = f"very good (above average, σ={confidence_std:.2f})"
            elif avg_confidence >= 0.65:
                analysis["overall_confidence"] = f"good (average performance, σ={confidence_std:.2f})"
            elif avg_confidence >= 0.55:
                analysis["overall_confidence"] = f"fair (below average, σ={confidence_std:.2f})"
            else:
                analysis["overall_confidence"] = f"needs improvement (bottom 25%, σ={confidence_std:.2f})"
            
            # Confidence stability with statistical interpretation
            if confidence_std < 0.08:
                analysis["confidence_stability"] = f"very stable (CV={confidence_std/avg_confidence:.2f})"
            elif confidence_std < 0.15:
                analysis["confidence_stability"] = f"mostly stable (CV={confidence_std/avg_confidence:.2f})"
            else:
                analysis["confidence_stability"] = f"variable (CV={confidence_std/avg_confidence:.2f})"
        
        # ENHANCED PACE AND TIMING ANALYSIS
        if word_timing_data:
            durations = [w["duration"] for w in word_timing_data]
            avg_word_duration = sum(durations) / len(durations)
            word_duration_std = (sum((d - avg_word_duration)**2 for d in durations) / len(durations))**0.5
            
            # Calculate words per minute
            if total_speech_time > 0:
                wpm = (total_words / total_speech_time) * 60
                analysis["pace_score"] = min(5.0, max(0.0, 3.0 + (160 - abs(wpm - 160)) / 40))  # Optimal around 160 WPM
                
                if wpm < 120:
                    pace_assessment = f"slow ({wpm:.0f} WPM) - increase pace"
                elif wpm > 200:
                    pace_assessment = f"fast ({wpm:.0f} WPM) - slow down for clarity"
                else:
                    pace_assessment = f"excellent pace ({wpm:.0f} WPM)"
            else:
                pace_assessment = "unable to calculate"
                
            # Rhythm consistency analysis
            rhythm_cv = word_duration_std / avg_word_duration if avg_word_duration > 0 else 1.0
            if rhythm_cv < 0.3:
                analysis["speech_rhythm"] = f"very consistent rhythm (CV={rhythm_cv:.2f})"
            elif rhythm_cv < 0.5:
                analysis["speech_rhythm"] = f"good rhythm (CV={rhythm_cv:.2f})"
            else:
                analysis["speech_rhythm"] = f"irregular rhythm (CV={rhythm_cv:.2f}) - practice timing"
                
            analysis["pace_analysis"] = pace_assessment
        
        # ADVANCED PAUSE ANALYSIS
        if pause_data:
            pause_durations = [p["duration"] for p in pause_data]
            avg_pause = sum(pause_durations) / len(pause_durations)
            long_pauses = sum(1 for p in pause_data if p["significance"] == "high")
            pause_frequency = len(pause_data) / total_speech_time if total_speech_time > 0 else 0
            
            # Categorize pause patterns
            strategic_pauses = sum(1 for p in pause_data if 0.3 <= p["duration"] <= 0.8)
            hesitation_pauses = sum(1 for p in pause_data if p["duration"] > 1.0)
            
            pause_score = 5.0
            if hesitation_pauses > len(pause_data) * 0.3:
                pause_score -= 2.0
            if pause_frequency > 2.0:  # More than 2 pauses per second indicates excessive hesitation
                pause_score -= 1.0
            if strategic_pauses > len(pause_data) * 0.4:
                pause_score += 0.5  # Bonus for strategic pauses
                
            analysis["pause_patterns"] = f"avg={avg_pause:.2f}s, strategic={strategic_pauses}, hesitation={hesitation_pauses}"
        
        # COMPREHENSIVE FILLER ANALYSIS
        if total_words > 0:
            total_fillers = sum(filler_counts.values())
            filler_rate = (total_fillers / total_words) * 100
            
            # Category-specific analysis
            filler_breakdown = []
            for category, count in filler_counts.items():
                if count > 0:
                    rate = (count / total_words) * 100
                    filler_breakdown.append(f"{category}:{rate:.1f}%")
            
            # Calculate filler clustering (fillers appearing in groups)
            filler_clusters = 0
            if len(filler_positions) >= 2:
                for i in range(1, len(filler_positions)):
                    if filler_positions[i]["timestamp"] - filler_positions[i-1]["timestamp"] < 3.0:
                        filler_clusters += 1
            
            cluster_rate = filler_clusters / len(filler_positions) if filler_positions else 0
            
            # Enhanced filler assessment
            if filler_rate < 2:
                filler_assessment = f"excellent fluency ({filler_rate:.1f}% fillers)"
            elif filler_rate < 5:
                filler_assessment = f"good fluency ({filler_rate:.1f}% fillers, minor improvement needed)"
            elif filler_rate < 10:
                filler_assessment = f"moderate fluency ({filler_rate:.1f}% fillers, practice needed)"
            else:
                filler_assessment = f"poor fluency ({filler_rate:.1f}% fillers, significant practice needed)"
            
            if cluster_rate > 0.3:
                filler_assessment += f" | clustering detected ({cluster_rate:.1f})"
                
            analysis["filler_analysis"] = filler_assessment
            if filler_breakdown:
                analysis["filler_analysis"] += f" | breakdown: {', '.join(filler_breakdown)}"
        
        # COMPREHENSIVE FLUENCY AND PERFORMANCE SCORING
        fluency_components = []
        
        # Confidence component (40% weight)
        if "confidence_score" in analysis:
            conf_score = analysis["confidence_score"] * 5  # Convert to 1-5 scale
            fluency_components.append(("confidence", conf_score, 0.4))
        
        # Pace component (25% weight)
        if "pace_score" in analysis:
            fluency_components.append(("pace", analysis["pace_score"], 0.25))
        
        # Filler component (20% weight) - inverse scoring
        if total_words > 0:
            total_fillers = sum(filler_counts.values())
            filler_rate = (total_fillers / total_words) * 100
            filler_score = max(0.0, 5.0 - (filler_rate / 2))  # allow 0 when fillers are high
            fluency_components.append(("fillers", filler_score, 0.2))
        
        # Consistency component (15% weight)
        if confidence_data:
            consistency_score = max(0.0, 5.0 - (confidence_std * 10))  # Lower std = higher score
            fluency_components.append(("consistency", consistency_score, 0.15))
        
        # Calculate weighted fluency score
        if fluency_components:
            weighted_sum = sum(score * weight for _, score, weight in fluency_components)
            total_weight = sum(weight for _, _, weight in fluency_components)
            analysis["fluency_score"] = weighted_sum / total_weight if total_weight > 0 else 0
            
            # Fluency interpretation with component breakdown
            score = analysis["fluency_score"]
            component_details = [f"{name}:{score:.1f}" for name, score, _ in fluency_components]
            
            if score >= 4.5:
                analysis["speaking_fluency"] = f"exceptional fluency ({score:.1f}/5.0) | {', '.join(component_details)}"
            elif score >= 4.0:
                analysis["speaking_fluency"] = f"excellent fluency ({score:.1f}/5.0) | {', '.join(component_details)}"
            elif score >= 3.5:
                analysis["speaking_fluency"] = f"very good fluency ({score:.1f}/5.0) | {', '.join(component_details)}"
            elif score >= 3.0:
                analysis["speaking_fluency"] = f"good fluency ({score:.1f}/5.0) | {', '.join(component_details)}"
            elif score >= 2.5:
                analysis["speaking_fluency"] = f"fair fluency ({score:.1f}/5.0) | {', '.join(component_details)}"
            else:
                analysis["speaking_fluency"] = f"poor fluency ({score:.1f}/5.0) | {', '.join(component_details)}"
        
        # PERFORMANCE INDICATORS AND IMPROVEMENT METRICS
        performance_flags = []
        improvement_suggestions = []
        
        if confidence_data and avg_confidence < 0.6:
            performance_flags.append("low_confidence")
            improvement_suggestions.append("practice content to increase certainty")
            
        if total_words > 0:
            total_fillers = sum(filler_counts.values())
            if (total_fillers / total_words) > 0.08:
                performance_flags.append("excessive_fillers")
                improvement_suggestions.append("practice reducing filler words")
                
        if pause_data:
            hesitation_pauses = sum(1 for p in pause_data if p["duration"] > 1.0)
            if hesitation_pauses > len(pause_data) * 0.3:
                performance_flags.append("frequent_hesitation")
                improvement_suggestions.append("improve content preparation")
        
        analysis["performance_indicators"] = ", ".join(performance_flags) if performance_flags else "strong_performance"
        analysis["improvement_metrics"] = " | ".join(improvement_suggestions) if improvement_suggestions else "maintain_current_level"
        
        # COMPARATIVE ANALYSIS (benchmarking against typical performance)
        benchmark_scores = {
            "beginner": {"confidence": 0.5, "fluency": 2.5, "pace": 2.0},
            "intermediate": {"confidence": 0.7, "fluency": 3.5, "pace": 3.5},
            "advanced": {"confidence": 0.8, "fluency": 4.2, "pace": 4.0},
            "expert": {"confidence": 0.9, "fluency": 4.8, "pace": 4.5}
        }
        
        current_metrics = {
            "confidence": analysis.get("confidence_score", 0),
            "fluency": analysis.get("fluency_score", 0),
            "pace": analysis.get("pace_score", 0)
        }
        
        best_match = "beginner"
        for level, benchmarks in benchmark_scores.items():
            if all(current_metrics[metric] >= benchmarks[metric] * 0.9 for metric in benchmarks):
                best_match = level
                
        analysis["comparative_analysis"] = f"performance_level: {best_match}"
        
        # PRONUNCIATION CLARITY ASSESSMENT
        pronunciation_score = 0.0
        pronunciation_assessment = "analysis unavailable"
        
        if words_data and confidence_data:
            # Calculate pronunciation clarity based on multiple factors
            
            # 1. Word-level confidence (40% weight)
            word_confidences = [word.get("probability", 0) for word in words_data if word.get("probability")]
            if word_confidences:
                avg_word_confidence = sum(word_confidences) / len(word_confidences)
                word_confidence_score = avg_word_confidence * 5  # Convert to 1-5 scale
            else:
                word_confidence_score = 2.5  # Default middle score
            
            # 2. Confidence consistency (25% weight)
            if len(word_confidences) > 1:
                confidence_variance = statistics.variance(word_confidences)
                consistency_score = max(0.0, 5.0 - (confidence_variance * 20))  # Lower variance = higher score
            else:
                consistency_score = 3.0
            
            # 3. Word recognition rate (20% weight)
            recognized_words = sum(1 for word in words_data if word.get("probability", 0) > 0.3)
            total_words = len(words_data)
            if total_words > 0:
                recognition_rate = recognized_words / total_words
                recognition_score = recognition_rate * 5
            else:
                recognition_score = 2.5
            
            # 4. Complex word handling (15% weight)
            # Identify potentially complex words (longer words, technical terms)
            complex_words = [word for word in words_data if len(word.get("word", "")) > 8]
            if complex_words:
                complex_word_confidences = [word.get("probability", 0) for word in complex_words]
                avg_complex_confidence = sum(complex_word_confidences) / len(complex_word_confidences)
                complex_word_score = avg_complex_confidence * 5
            else:
                complex_word_score = 3.0  # No complex words to assess
            
            # Calculate weighted pronunciation score
            pronunciation_score = (
                word_confidence_score * 0.4 +
                consistency_score * 0.25 +
                recognition_score * 0.2 +
                complex_word_score * 0.15
            )
            
            # Pronunciation assessment interpretation
            if pronunciation_score >= 4.5:
                pronunciation_assessment = f"exceptional pronunciation clarity ({pronunciation_score:.1f}/5.0) | high word recognition, consistent confidence"
            elif pronunciation_score >= 4.0:
                pronunciation_assessment = f"excellent pronunciation clarity ({pronunciation_score:.1f}/5.0) | strong word recognition, good consistency"
            elif pronunciation_score >= 3.5:
                pronunciation_assessment = f"very good pronunciation clarity ({pronunciation_score:.1f}/5.0) | good word recognition, stable confidence"
            elif pronunciation_score >= 3.0:
                pronunciation_assessment = f"good pronunciation clarity ({pronunciation_score:.1f}/5.0) | adequate word recognition, some variation"
            elif pronunciation_score >= 2.5:
                pronunciation_assessment = f"fair pronunciation clarity ({pronunciation_score:.1f}/5.0) | moderate word recognition, some uncertainty"
            else:
                pronunciation_assessment = f"poor pronunciation clarity ({pronunciation_score:.1f}/5.0) | low word recognition, inconsistent confidence"
            
            # Add detailed breakdown
            pronunciation_assessment += f" | breakdown: word_conf:{word_confidence_score:.1f}, consistency:{consistency_score:.1f}, recognition:{recognition_score:.1f}, complex:{complex_word_score:.1f}"
        
        analysis["pronunciation_clarity"] = pronunciation_assessment
        analysis["clarity_score"] = pronunciation_score
        
        logger.info(f"Enhanced Whisper analysis completed: confidence={analysis.get('confidence_score', 0):.2f}, fluency={analysis.get('fluency_score', 0):.2f}, pronunciation={pronunciation_score:.2f}")
        return analysis
        
    except Exception as e:
        logger.warning(f"Error analyzing Whisper metrics: {e}")
        return {
            "overall_confidence": "analysis unavailable",
            "speaking_fluency": "analysis unavailable", 
            "word_certainty": "analysis unavailable",
            "speech_consistency": "analysis unavailable",
            "pace_stability": "analysis unavailable",
            "filler_frequency": "analysis unavailable",
            "pronunciation_clarity": "analysis unavailable",
            "pause_analysis": "analysis unavailable",
            "confidence_distribution": "analysis unavailable"
        }


def _trimmed_mean(values: List[float], lower: float = 0.1, upper: float = 0.9) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    lo_idx = int(math.floor(n * lower))
    hi_idx = int(math.ceil(n * upper))
    hi_idx = max(lo_idx + 1, min(hi_idx, n))
    trimmed = sorted_vals[lo_idx:hi_idx]
    return sum(trimmed) / len(trimmed) if trimmed else sum(sorted_vals) / n


def _median(values: List[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    m = len(s)
    mid = m // 2
    if m % 2 == 0:
        return (s[mid - 1] + s[mid]) / 2.0
    return s[mid]


def _mad(values: List[float]) -> float:
    if not values:
        return 0.0
    med = _median(values)
    deviations = [abs(v - med) for v in values]
    return _median(deviations)


def _std(values: List[float]) -> float:
    if not values:
        return 0.0
    m = sum(values) / len(values)
    return (sum((v - m) ** 2 for v in values) / len(values)) ** 0.5


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _map_inverse(value: float, good: float, bad: float, out_min: float = 0.0, out_max: float = 5.0) -> float:
    if value <= good:
        return out_max
    if value >= bad:
        return out_min
    ratio = (value - good) / (bad - good)
    return _clip(out_max * (1.0 - ratio), out_min, out_max)


def _band_score(x: float, opt_low: float, opt_high: float, min_x: float, max_x: float, out_min: float = 0.0, out_max: float = 5.0) -> float:
    if x <= min_x or x >= max_x:
        return out_min
    if opt_low <= x <= opt_high:
        return out_max
    if x < opt_low:
        return out_min + (x - min_x) / (opt_low - min_x) * (out_max - out_min)
    return out_min + (max_x - x) / (max_x - opt_high) * (out_max - out_min)


def _collect_words(whisper_result) -> Tuple[List[Dict], List[Dict]]:
    segments = whisper_result.get("segments", [])
    words: List[Dict] = []
    seg_meta: List[Dict] = []
    for idx, seg in enumerate(segments):
        seg_meta.append({
            "idx": idx,
            "avg_logprob": seg.get("avg_logprob", -1.0)
        })
        for widx, w in enumerate(seg.get("words", [])):
            words.append({
                "text": (w.get("word", "").strip() or ""),
                "start": w.get("start", 0.0),
                "end": w.get("end", 0.0),
                "prob": w.get("probability", None),
                "seg_idx": idx,
                "widx": widx,
            })
    return words, seg_meta


def _derive_prob_from_logprob(avg_logprob: float) -> float:
    # Map Whisper avg_logprob from ~[-3.0, 0.0] to [0,1]
    low, high = -3.0, -0.05
    return _clip((avg_logprob - low) / (high - low), 0.0, 1.0)


def _normalize_word_confidences(words: List[Dict], seg_meta: List[Dict]) -> List[float]:
    seg_idx_to_prob = {s["idx"]: _derive_prob_from_logprob(s["avg_logprob"]) for s in seg_meta}
    probs: List[float] = []
    for w in words:
        p = w.get("prob")
        if p is None:
            p = seg_idx_to_prob.get(w["seg_idx"], 0.5)
        probs.append(_clip(float(p), 0.0, 1.0))
    return probs


def _inter_word_gaps(words: List[Dict]) -> List[float]:
    gaps: List[float] = []
    for i in range(1, len(words)):
        prev_end = float(words[i - 1].get("end", 0.0))
        cur_start = float(words[i].get("start", 0.0))
        gap = cur_start - prev_end
        if 0.05 <= gap <= 1.0:
            gaps.append(gap)
    return gaps


def _detect_fillers(words: List[Dict]) -> Tuple[int, Dict[str, int]]:
    # Boundary-aware detection for common fillers
    multi_fillers = {
        ("you", "know"): "hedge",
        ("i", "mean"): "hedge",
        ("kind", "of"): "hedge",
        ("sort", "of"): "hedge",
    }
    single_fillers = {
        "um": "basic",
        "uh": "basic",
        "er": "basic",
        "ah": "basic",
        "hmm": "basic",
        # "like" handled specially
    }

    total = 0
    breakdown: Dict[str, int] = {"basic": 0, "hedge": 0, "transition": 0, "emphasis": 0}
    i = 0
    n = len(words)
    while i < n:
        w = words[i]
        txt = w.get("text", "").strip().lower()
        # multi-word detection first
        if i + 1 < n:
            pair = (txt, words[i + 1].get("text", "").strip().lower())
            if pair in multi_fillers:
                breakdown[multi_fillers[pair]] += 1
                total += 1
                i += 2
                continue
        # single-word detection
        if txt in single_fillers:
            breakdown[single_fillers[txt]] += 1
            total += 1
            i += 1
            continue
        if txt == "like":
            # treat as filler only if surrounded by pauses >= 0.15s
            pre_gap = words[i]["start"] - (words[i - 1]["end"] if i > 0 else words[i]["start"]) if i > 0 else 0.0
            post_gap = (words[i + 1]["start"] - words[i]["end"]) if i + 1 < n else 0.0
            if pre_gap >= 0.15 and post_gap >= 0.15:
                breakdown["transition"] += 1
                total += 1
                i += 1
                continue
        i += 1
    return total, breakdown


def _detect_repetitions(words: List[Dict]) -> float:
    if len(words) < 2:
        return 0.0
    repeats = 0
    for i in range(1, len(words)):
        cur = words[i]
        prev = words[i - 1]
        if cur.get("text", "").strip().lower() == prev.get("text", "").strip().lower():
            if (cur.get("start", 0.0) - prev.get("end", 0.0)) < 0.5:
                repeats += 1
    return repeats / max(1, len(words))


def _frame_rms(samples: bytes, sample_width: int) -> float:
    if sample_width == 2:
        # 16-bit PCM
        import struct
        count = len(samples) // 2
        if count == 0:
            return 0.0
        unpacked = struct.unpack("<" + ("h" * count), samples)
        mean_sq = sum(x * x for x in unpacked) / count
        return math.sqrt(mean_sq) / 32768.0
    # Fallback: average absolute value normalized by max byte
    if not samples:
        return 0.0
    return sum(abs(b - 128) for b in samples) / (len(samples) * 128.0)


def _vad_and_snr(audio_path: str, frame_ms: int = 20) -> Tuple[float, float]:
    try:
        with wave.open(audio_path, "rb") as wf:
            sample_rate = wf.getframerate()
            nchannels = wf.getnchannels()
            sw = wf.getsampwidth()
            total_frames = wf.getnframes()
            frame_len = int(sample_rate * frame_ms / 1000)
            voiced_rms: List[float] = []
            unvoiced_rms: List[float] = []
            voiced_frames = 0
            total_time = total_frames / float(sample_rate)

            # Read frames channel-agnostic by taking first channel for simplicity
            # Process chunk by chunk
            while True:
                frames = wf.readframes(frame_len)
                if not frames:
                    break
                # If stereo, take first channel
                if nchannels > 1 and sw == 2:
                    import array
                    arr = array.array('h', frames)
                    frames = arr[0::nchannels].tobytes()

                rms = _frame_rms(frames, sw)
                unvoiced_rms.append(rms)

            if not unvoiced_rms or total_time <= 0:
                return 0.0, 0.0

            # Determine threshold at 60th percentile for voiced
            sorted_rms = sorted(unvoiced_rms)
            thr = sorted_rms[int(0.60 * len(sorted_rms))]
            # Rewind and classify voiced frames
            wf.rewind()
            while True:
                frames = wf.readframes(frame_len)
                if not frames:
                    break
                if nchannels > 1 and sw == 2:
                    import array
                    arr = array.array('h', frames)
                    frames = arr[0::nchannels].tobytes()
                rms = _frame_rms(frames, sw)
                if rms >= thr:
                    voiced_rms.append(rms)
                    voiced_frames += 1
                else:
                    unvoiced_rms.append(rms)

            speech_seconds = voiced_frames * (frame_ms / 1000.0)
            if not voiced_rms:
                return 0.0, 0.0

            # SNR estimate using medians for robustness
            speech_level = _median(voiced_rms)
            if unvoiced_rms:
                noise_level = _median(sorted(unvoiced_rms)[: max(1, int(0.2 * len(unvoiced_rms)))])
            else:
                noise_level = max(1e-6, 0.001)
            noise_level = max(noise_level, 1e-6)
            snr = 20.0 * math.log10(max(speech_level, noise_level + 1e-6) / noise_level)
            return speech_seconds, snr
    except Exception:
        return 0.0, 0.0


def _map_snr_to_5(snr_db: float) -> float:
    # Piecewise linear: 0->1, 5->2, 10->3, 20->4, 30->5
    points = [(0, 1), (5, 2), (10, 3), (20, 4), (30, 5)]
    if snr_db <= points[0][0]:
        return points[0][1]
    for (x1, y1), (x2, y2) in zip(points, points[1:]):
        if x1 <= snr_db <= x2:
            t = (snr_db - x1) / (x2 - x1)
            return y1 + t * (y2 - y1)
    return points[-1][1]


def analyze_whisper_metrics_v2(whisper_result, audio_path: Optional[str] = None) -> dict:
    try:
        words, seg_meta = _collect_words(whisper_result)
        confidences = _normalize_word_confidences(words, seg_meta)
        total_words = len(words)

        # Confidence metrics (robust)
        conf_trim = _trimmed_mean(confidences, 0.1, 0.9)
        conf_mad = _mad(confidences)
        conf_score = _clip(conf_trim, 0.0, 1.0)

        # Stability label
        if conf_mad < 0.05:
            stability = "very stable"
        elif conf_mad < 0.10:
            stability = "mostly stable"
        else:
            stability = "variable"

        # Percentiles and trend
        conf_sorted = sorted(confidences)
        if conf_sorted:
            n = len(conf_sorted)
            p25 = conf_sorted[int(0.25 * n)]
            p50 = conf_sorted[int(0.50 * n)]
            p75 = conf_sorted[int(0.75 * n)]
            p90 = conf_sorted[int(0.90 * n)]
            conf_percentiles = f"P25:{p25:.2f} P50:{p50:.2f} P75:{p75:.2f} P90:{p90:.2f}"
        else:
            conf_percentiles = "unknown"

        if len(confidences) >= 3:
            third = max(1, len(confidences) // 3)
            first_third = sum(confidences[:third]) / third
            last_third = sum(confidences[-third:]) / third
            trend_delta = last_third - first_third
            if trend_delta > 0.1:
                conf_trend = f"improving (+{trend_delta:.2f})"
            elif trend_delta < -0.1:
                conf_trend = f"declining ({trend_delta:.2f})"
            else:
                conf_trend = "stable"
        else:
            conf_trend = "unknown"

        # Timing smoothness
        gaps = _inter_word_gaps(words)
        if gaps and sum(gaps) > 0:
            gaps_mean = sum(gaps) / len(gaps)
            gaps_std = _std(gaps)
            cv = gaps_std / gaps_mean if gaps_mean > 0 else 1.0
        else:
            cv = 1.0
        smoothness_score = _map_inverse(cv, good=0.25, bad=0.6, out_min=0.0, out_max=5.0)

        # Pace via simple VAD on audio_path if provided
        if audio_path:
            speech_seconds, snr_db = _vad_and_snr(audio_path)
        else:
            speech_seconds, snr_db = 0.0, 0.0

        if speech_seconds > 0 and total_words > 0:
            wpm = (total_words / speech_seconds) * 60.0
            pace_score = _band_score(wpm, opt_low=140.0, opt_high=170.0, min_x=90.0, max_x=210.0, out_min=0.0, out_max=5.0)
            if wpm < 120:
                pace_text = f"slow ({wpm:.0f} WPM)"
            elif wpm > 200:
                pace_text = f"fast ({wpm:.0f} WPM)"
            else:
                pace_text = f"excellent pace ({wpm:.0f} WPM)"
        else:
            wpm = 0.0
            pace_score = 0.0
            pace_text = "unable to calculate"

        # Fillers and repetitions
        filler_total, filler_breakdown = _detect_fillers(words)
        filler_rate = (filler_total / total_words) if total_words > 0 else 0.0
        filler_score = _map_inverse(filler_rate, good=0.02, bad=0.10, out_min=0.0, out_max=5.0)
        repetition_ratio = _detect_repetitions(words)
        repetition_score = _map_inverse(repetition_ratio, good=0.01, bad=0.05, out_min=0.0, out_max=5.0)

        fluency_score = (
            0.35 * smoothness_score + 0.30 * pace_score + 0.25 * filler_score + 0.10 * repetition_score
        )

        if fluency_score >= 4.5:
            fluency_text = f"exceptional fluency ({fluency_score:.1f}/5.0)"
        elif fluency_score >= 4.0:
            fluency_text = f"excellent fluency ({fluency_score:.1f}/5.0)"
        elif fluency_score >= 3.5:
            fluency_text = f"very good fluency ({fluency_score:.1f}/5.0)"
        elif fluency_score >= 3.0:
            fluency_text = f"good fluency ({fluency_score:.1f}/5.0)"
        elif fluency_score >= 2.5:
            fluency_text = f"fair fluency ({fluency_score:.1f}/5.0)"
        else:
            fluency_text = f"poor fluency ({fluency_score:.1f}/5.0)"

        # Pronunciation clarity
        long_word_confs = [c for c, w in zip(confidences, words) if len(w.get("text", "").strip()) >= 6]
        long_conf = _trimmed_mean(long_word_confs, 0.1, 0.9) if long_word_confs else conf_score
        if long_word_confs:
            var_long = _std(long_word_confs)
        else:
            var_long = _std(confidences)
        consistency_component = _map_inverse(var_long, good=0.10, bad=0.25, out_min=0.0, out_max=5.0)
        snr_component = _map_snr_to_5(snr_db)
        short_low_ratio = 0.0
        if total_words > 0:
            short_low = sum(1 for c, w in zip(confidences, words) if len(w.get("text", "")) < 3 and c < 0.5)
            short_low_ratio = short_low / total_words
        short_word_component = _map_inverse(short_low_ratio, good=0.02, bad=0.12, out_min=0.0, out_max=5.0)

        clarity_score = 0.40 * (long_conf * 5.0) + 0.25 * consistency_component + 0.25 * snr_component + 0.10 * short_word_component
        if clarity_score >= 4.5:
            pronunciation_text = f"exceptional pronunciation clarity ({clarity_score:.1f}/5.0)"
        elif clarity_score >= 4.0:
            pronunciation_text = f"excellent pronunciation clarity ({clarity_score:.1f}/5.0)"
        elif clarity_score >= 3.5:
            pronunciation_text = f"very good pronunciation clarity ({clarity_score:.1f}/5.0)"
        elif clarity_score >= 3.0:
            pronunciation_text = f"good pronunciation clarity ({clarity_score:.1f}/5.0)"
        elif clarity_score >= 2.5:
            pronunciation_text = f"fair pronunciation clarity ({clarity_score:.1f}/5.0)"
        else:
            pronunciation_text = f"poor pronunciation clarity ({clarity_score:.1f}/5.0)"

        # Overall confidence label
        if conf_score >= 0.85:
            overall_conf_text = "excellent"
        elif conf_score >= 0.75:
            overall_conf_text = "very good"
        elif conf_score >= 0.65:
            overall_conf_text = "good"
        elif conf_score >= 0.55:
            overall_conf_text = "fair"
        else:
            overall_conf_text = "needs improvement"

        # Performance flags / suggestions
        flags = []
        improvements = []
        if filler_rate > 0.08:
            flags.append("excessive_fillers")
            improvements.append("reduce filler words")
        if cv > 0.6:
            flags.append("irregular_rhythm")
            improvements.append("practice timing and pacing")
        if speech_seconds > 0 and (wpm < 120 or wpm > 200):
            flags.append("suboptimal_pace")
            improvements.append("adjust speaking pace")

        return {
            "overall_confidence": overall_conf_text,
            "confidence_score": round(conf_score, 2),
            "confidence_stability": stability,
            "confidence_trend": conf_trend,
            "confidence_percentiles": conf_percentiles,
            "speaking_fluency": fluency_text,
            "fluency_score": round(fluency_score, 2),
            "pace_analysis": pace_text,
            "pace_score": round(pace_score, 2),
            "speech_rhythm": f"cv={cv:.2f}",
            "pause_patterns": "n/a",  # simplified in v2 core
            "filler_analysis": f"rate={filler_rate*100:.1f}%",
            "pronunciation_clarity": pronunciation_text,
            "clarity_score": round(clarity_score, 2),
            "comparative_analysis": "performance_level: n/a",
            "performance_indicators": ", ".join(flags) if flags else "strong_performance",
            "improvement_metrics": " | ".join(improvements) if improvements else "maintain_current_level",
        }
    except Exception as e:
        logger.warning(f"Error in analyze_whisper_metrics_v2: {e}")
        return {
            "overall_confidence": "analysis unavailable",
            "speaking_fluency": "analysis unavailable",
            "pronunciation_clarity": "analysis unavailable",
        }


def format_transcription_for_user(whisper_result) -> str:
    """
    Format transcription for immediate display to user (clean, readable format)
    
    Args:
        whisper_result: Whisper transcription result with segments
        
    Returns:
        Clean, formatted transcription for user display
    """
    try:
        transcript_lines = []
        
        # Add clean header
        transcript_lines.append("**🎤 Your Transcription:**\n")
        
        # Get segments with timestamps
        segments = whisper_result.get("segments", [])
        
        if not segments:
            # Fallback to simple text if no segments
            return "**🎤 Your Transcription:**\n\n" + whisper_result.get("text", "").strip()
        
        for i, segment in enumerate(segments, 1):
            start_time = segment.get("start", 0)
            text = segment.get("text", "").strip()
            
            if text:
                # Format timestamp as MM:SS
                start_min = int(start_time // 60)
                start_sec = int(start_time % 60)
                timestamp = f"{start_min:02d}:{start_sec:02d}"
                
                # Create clean line with timestamp and text
                line = f"**[{timestamp}]** {text}"
                transcript_lines.append(line)
        
        if len(transcript_lines) > 1:
            formatted = "\n".join(transcript_lines)
            logger.info(f"Formatted user transcription with {len(transcript_lines)-1} segments")
            return formatted
        else:
            # Fallback to simple text
            return "**🎤 Your Transcription:**\n\n" + whisper_result.get("text", "").strip()
            
    except Exception as e:
        logger.warning(f"Error formatting user transcription: {e}")
        # Fallback to simple text
        return "**🎤 Your Transcription:**\n\n" + whisper_result.get("text", "").strip()


def format_transcript_for_feedback(whisper_result, audio_analysis=None, whisper_metrics=None) -> str:
    """
    Format Whisper transcription result with line numbers, timestamps, and comprehensive analysis
    
    Args:
        whisper_result: Whisper transcription result with segments
        audio_analysis: Dictionary with audio feature analysis
        whisper_metrics: Dictionary with Whisper-based confidence and fluency analysis
        
    Returns:
        Formatted transcript with line numbers, timestamps, and delivery insights
    """
    try:
        transcript_lines = []
        
        # Add comprehensive analysis header
        analysis_header = "\n**COMPREHENSIVE SPEECH ANALYSIS:**\n"
        
        # Enhanced Whisper-based metrics with comprehensive analysis
        if whisper_metrics:
            analysis_header += "\n*📊 PERFORMANCE SCORES:*\n"
            analysis_header += f"- Confidence Score: {whisper_metrics.get('confidence_score', 0):.2f}/1.0\n"
            analysis_header += f"- Fluency Score: {whisper_metrics.get('fluency_score', 0):.2f}/5.0\n"
            analysis_header += f"- Pace Score: {whisper_metrics.get('pace_score', 0):.2f}/5.0\n"
            analysis_header += f"- Clarity Score: {whisper_metrics.get('clarity_score', 0):.2f}/5.0\n"
            
            analysis_header += "\n*🎯 DETAILED ANALYSIS:*\n"
            analysis_header += f"- Overall Confidence: {whisper_metrics.get('overall_confidence', 'unknown')}\n"
            analysis_header += f"- Speaking Fluency: {whisper_metrics.get('speaking_fluency', 'unknown')}\n"
            analysis_header += f"- Confidence Stability: {whisper_metrics.get('confidence_stability', 'unknown')}\n"
            analysis_header += f"- Confidence Trend: {whisper_metrics.get('confidence_trend', 'unknown')}\n"
            
            analysis_header += "\n*⏱️ PACE & TIMING:*\n"
            analysis_header += f"- Pace Analysis: {whisper_metrics.get('pace_analysis', 'unknown')}\n"
            analysis_header += f"- Speech Rhythm: {whisper_metrics.get('speech_rhythm', 'unknown')}\n"
            analysis_header += f"- Pause Patterns: {whisper_metrics.get('pause_patterns', 'unknown')}\n"
            
            analysis_header += "\n*💬 FLUENCY & CLARITY:*\n"
            analysis_header += f"- Filler Analysis: {whisper_metrics.get('filler_analysis', 'unknown')}\n"
            analysis_header += f"- Pronunciation Clarity: {whisper_metrics.get('pronunciation_clarity', 'unknown')}\n"
            analysis_header += f"- Vocabulary Confidence: {whisper_metrics.get('vocabulary_confidence', 'unknown')}\n"
            
            analysis_header += "\n*📈 STATISTICAL INSIGHTS:*\n"
            analysis_header += f"- Confidence Percentiles: {whisper_metrics.get('confidence_percentiles', 'unknown')}\n"
            analysis_header += f"- Performance Level: {whisper_metrics.get('comparative_analysis', 'unknown')}\n"
            analysis_header += f"- Performance Flags: {whisper_metrics.get('performance_indicators', 'unknown')}\n"
            analysis_header += f"- Improvement Areas: {whisper_metrics.get('improvement_metrics', 'unknown')}\n"
        
        # Audio signal analysis (supplementary)
        if audio_analysis:
            analysis_header += "\n*Audio Signal Analysis:*\n"
            analysis_header += f"- Speaking Rate: {audio_analysis.get('speaking_rate', 'unknown')}\n"
            analysis_header += f"- Volume Consistency: {audio_analysis.get('volume_consistency', 'unknown')}\n"
            analysis_header += f"- Energy Level: {audio_analysis.get('energy_level', 'unknown')}\n"
            analysis_header += f"- Pace Assessment: {audio_analysis.get('pace_assessment', 'unknown')}\n"
        
        analysis_header += "\n**TRANSCRIPT:**\n"
        transcript_lines.append(analysis_header)
        
        # Get segments with timestamps
        segments = whisper_result.get("segments", [])
        
        if not segments:
            # Fallback to simple text if no segments
            base_text = whisper_result.get("text", "").strip()
            if audio_analysis:
                return analysis_header + base_text
            return base_text
        
        for i, segment in enumerate(segments, 1):
            start_time = segment.get("start", 0)
            end_time = segment.get("end", 0)
            text = segment.get("text", "").strip()
            
            if text:
                # Format timestamp as MM:SS
                start_min = int(start_time // 60)
                start_sec = int(start_time % 60)
                timestamp = f"{start_min:02d}:{start_sec:02d}"
                
                # Create line with number, timestamp, and text
                line = f"Line {i} [{timestamp}]: {text}"
                transcript_lines.append(line)
        
        if len(transcript_lines) > (1 if audio_analysis else 0):
            formatted = "\n".join(transcript_lines)
            logger.info(f"Formatted transcript with {len(transcript_lines)} lines (including audio analysis)")
            return formatted
        else:
            # Fallback to simple text
            base_text = whisper_result.get("text", "").strip()
            if audio_analysis:
                return transcript_lines[0] + base_text  # Include analysis header
            return base_text
            
    except Exception as e:
        logger.warning(f"Error formatting transcript: {e}")
        # Fallback to simple text
        return whisper_result.get("text", "").strip()


def get_whisper_model():
    """
    Get or load the Whisper model (cached for performance)
    
    Returns:
        Loaded Whisper model
    """
    global _whisper_model
    
    if _whisper_model is None:
        try:
            import whisper
            import ssl
            import urllib.request
            
            logger.info("Loading Whisper model (this may take a moment on first run)...")
            
            # Configure SSL to handle certificate issues
            try:
                # Create an SSL context that doesn't verify certificates
                ssl_context = ssl.create_default_context()
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE
                
                # Apply the SSL context globally for urllib
                opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ssl_context))
                urllib.request.install_opener(opener)
                
                logger.info("SSL context configured for model download")
            except Exception as ssl_error:
                logger.warning(f"Could not configure SSL context: {ssl_error}")
            
            # Import whisper here to avoid import errors if not installed
            import whisper
            
            # Use 'base' model for good balance of speed vs accuracy
            # Options: tiny, base, small, medium, large
            _whisper_model = whisper.load_model("base")
            logger.info("Whisper model loaded successfully")
        except ImportError:
            logger.error("Whisper package not found. Please install: pip install openai-whisper")
            raise
        except Exception as e:
            logger.error(f"Error loading Whisper model: {e}")
            raise
    
    return _whisper_model


def quick_transcribe_chunk(audio_data: bytes) -> str:
    """
    Fast transcription for short audio chunks. Returns plain text only.
    Optimized for latency: minimal options, no extra analysis.
    """
    try:
        model = get_whisper_model()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_file.write(audio_data)
            temp_file.flush()
            temp_audio_path = temp_file.name
        try:
            result = model.transcribe(
                temp_audio_path,
                language="en",
                task="transcribe",
                verbose=False,
                word_timestamps=False,
                temperature=0.0,
                beam_size=1,
                best_of=1,
                condition_on_previous_text=False,
                no_speech_threshold=0.8,
            )
            text = (result.get("text") or "").strip()
            return text
        finally:
            try:
                os.remove(temp_audio_path)
            except Exception:
                pass
    except Exception as exc:
        logger.warning(f"quick_transcribe_chunk failed: {exc}")
        return ""

def transcribe_audio_with_whisper(audio_data: bytes) -> str:
    """
    Transcribe audio using Local Whisper model
    
    Args:
        audio_data: Audio data in bytes (WAV format)
        
    Returns:
        Transcribed text
    """
    try:
        logger.info(f"Starting audio transcription. Audio data size: {len(audio_data)} bytes")
        
        # Get the Whisper model
        logger.info("Getting Whisper model...")
        model = get_whisper_model()
        logger.info("Whisper model ready")
        
        # Create a temporary file for the audio data
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_file.write(audio_data)
            temp_file.flush()
            temp_audio_path = temp_file.name
            logger.info(f"Created temporary audio file: {temp_audio_path}, size: {os.path.getsize(temp_audio_path)} bytes")
        
        try:
            # Add audio analysis
            try:
                import wave
                with wave.open(temp_audio_path, 'rb') as wav_file:
                    frames = wav_file.getnframes()
                    sample_rate = wav_file.getframerate()
                    duration = frames / sample_rate
                    channels = wav_file.getnchannels()
                    sample_width = wav_file.getsampwidth()
                    logger.info(f"Audio analysis: {duration:.2f}s duration, {sample_rate}Hz, {channels} channels, {sample_width} bytes/sample")
            except Exception as audio_error:
                logger.warning(f"Could not analyze audio file: {audio_error}")
            
            # Transcribe the audio with comprehensive analysis
            logger.info("Starting Whisper transcription with comprehensive analysis...")
            result = model.transcribe(
                temp_audio_path,
                language="en",  # Force English
                task="transcribe",  # Explicit task
                verbose=False,  # Reduce verbosity
                word_timestamps=True,  # Enable word-level timestamps
                temperature=0.0,  # Deterministic for consistent confidence scores
                beam_size=5,  # Use beam search for better accuracy
                best_of=1,  # Deterministic selection
                condition_on_previous_text=True,  # Better context
                compression_ratio_threshold=2.4,  # Detect hallucination
                logprob_threshold=-1.0,  # Confidence threshold
                no_speech_threshold=0.6  # Silence detection
            )
            
            transcript = result["text"].strip()
            logger.info(f"Transcription completed: '{transcript}' ({len(transcript)} characters)")
            
            if not transcript:
                logger.warning("Transcription result is empty - audio may be silent, too quiet, or in wrong format")
                return "Audio transcription completed but no speech was detected. Please check audio volume and clarity."
            
            # Analyze audio features for delivery assessment
            audio_analysis = analyze_audio_features(temp_audio_path)
            
            # Show transcription immediately to user
            user_transcription = format_transcription_for_user(result)

            # Analyze Whisper's output metrics using v2 (default)
            whisper_metrics = analyze_whisper_metrics_v2(result, temp_audio_path)

            # Optional: rubric + tips (content/structure via LLM, clarity/delivery via metrics)
            rubric_and_tips = None
            try:
                from api.utils.feedback import generate_rubric_and_tips_sync
                rubric_and_tips = generate_rubric_and_tips_sync(result, whisper_metrics)
            except Exception as _:
                rubric_and_tips = None

            # Format comprehensive analysis for AI
            formatted_transcript = format_transcript_for_feedback(result, audio_analysis, whisper_metrics)

            payload = {
                "user_transcription": user_transcription,
                "ai_analysis": formatted_transcript,
            }
            if rubric_and_tips is not None:
                payload["rubric"] = {
                    "content": rubric_and_tips.content,
                    "structure": rubric_and_tips.structure,
                    "clarity": rubric_and_tips.clarity,
                    "delivery": rubric_and_tips.delivery,
                }
                payload["tips"] = [t.model_dump() for t in rubric_and_tips.tips]

            return payload
            
        finally:
            # Clean up the temporary file
            try:
                os.unlink(temp_audio_path)
                logger.info("Temporary audio file cleaned up")
            except OSError as cleanup_error:
                logger.warning(f"Could not clean up temporary file: {cleanup_error}")
                
    except Exception as e:
        logger.error(f"Error transcribing audio with Local Whisper: {e}", exc_info=True)
        # Return a more informative fallback message
        return f"Audio transcription failed due to technical error: {str(e)[:100]}..."
