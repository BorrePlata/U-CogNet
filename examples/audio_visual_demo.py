#!/usr/bin/env python3
# U-CogNet Audio-Visual Synthesis Example
# Demonstration of the universal audio-visual perception system

import asyncio
import numpy as np
import logging
from typing import Dict, Any

# Import the audio-visual module
from ucognet.modules.audio import (
    AudioVisualSynthesizer,
    LibrosaFeatureExtractor,
    CognitiveAudioPerception,
    ArtisticVisualExpression,
    ArtisticVisualRenderer,
    AudioVisualEvaluator
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AudioVisualDemo:
    """Demonstration of the audio-visual synthesis system."""

    def __init__(self):
        self.synthesizer = AudioVisualSynthesizer()
        self.evaluator = AudioVisualEvaluator()

    async def initialize_system(self):
        """Initialize the complete audio-visual system."""
        logger.info("Initializing U-CogNet Audio-Visual System...")

        # Register all components
        self.synthesizer.register_feature_extractor(LibrosaFeatureExtractor())
        self.synthesizer.register_perception_engine(CognitiveAudioPerception())
        self.synthesizer.register_visual_expressor(ArtisticVisualExpression())
        self.synthesizer.register_visual_renderer(ArtisticVisualRenderer())

        # Configuration for high-quality synthesis
        config = {
            'quality_preset': 'high_quality',
            'real_time_processing': False,
            'enable_caching': True,
            'cache_size': 50,
            'max_concurrent_syntheses': 2,

            # Component-specific configurations
            'feature_extractor': {
                'fft_window_size': 2048,
                'hop_length': 512,
                'enable_mfcc': True,
                'enable_chroma': True
            },

            'perception_engine': {
                'emotion_sensitivity': 0.8,
                'context_awareness': 0.9
            },

            'visual_expressor': {
                'artistic_style': 'expressive',
                'color_vibrancy': 0.8,
                'symbol_density': 0.6
            },

            'visual_renderer': {
                'canvas_width': 1024,
                'canvas_height': 768,
                'antialiasing': True,
                'color_depth': 32
            }
        }

        # Initialize synthesizer
        await self.synthesizer.initialize(config)

        # Initialize evaluator
        await self.evaluator.initialize({
            'adaptation_strategy': 'balanced',
            'max_history_size': 200
        })

        logger.info("System initialization complete!")

    async def demonstrate_environmental_sounds(self):
        """Demonstrate synthesis of various environmental sounds."""
        logger.info("Demonstrating environmental sound synthesis...")

        # Simulated audio data for different environmental sounds
        test_sounds = {
            'birdsong': self._generate_birdsong_audio(),
            'explosion': self._generate_explosion_audio(),
            'alarm': self._generate_alarm_audio(),
            'nature': self._generate_nature_audio(),
            'urban': self._generate_urban_audio()
        }

        results = []

        for sound_type, audio_data in test_sounds.items():
            logger.info(f"Synthesizing {sound_type}...")

            # Synthesis context
            context = {
                'sound_type': sound_type,
                'environment': 'demonstration',
                'output_format': 'image',
                'quality_requirement': 'high'
            }

            # Perform synthesis
            result = await self.synthesizer.synthesize_audio_visual(audio_data, context)

            if result.rendered_visual:
                logger.info(f"✓ {sound_type} synthesis successful")
                logger.info(f"  Processing time: {result.processing_time:.2f}ms")
                logger.info(f"  Sound type detected: {result.perception.sound_type}")
                logger.info(f"  Emotional valence: {result.perception.emotional_valence:.2f}")
                logger.info(f"  Arousal level: {result.perception.arousal_level:.2f}")
                logger.info(f"  Visual style: {result.expression.style}")
                logger.info(f"  Visual intensity: {result.expression.intensity:.2f}")
            else:
                logger.error(f"✗ {sound_type} synthesis failed: {result.metadata.get('error', 'Unknown error')}")

            results.append(result)

        return results

    async def demonstrate_batch_processing(self):
        """Demonstrate batch processing capabilities."""
        logger.info("Demonstrating batch processing...")

        # Generate batch of audio samples
        batch_audio = [self._generate_random_audio() for _ in range(5)]
        batch_contexts = [
            {'batch_id': i, 'priority': 'normal', 'output_format': 'numpy'}
            for i in range(5)
        ]

        # Process batch
        batch_results = await self.synthesizer.synthesize_batch(batch_audio, batch_contexts)

        logger.info(f"Batch processing complete: {len(batch_results)} items processed")

        successful = sum(1 for r in batch_results if r.rendered_visual is not None)
        logger.info(f"Success rate: {successful}/{len(batch_results)}")

        return batch_results

    async def demonstrate_evaluation_and_adaptation(self, synthesis_results):
        """Demonstrate evaluation and adaptation capabilities."""
        logger.info("Demonstrating evaluation and adaptation...")

        evaluations = []

        # Evaluate each synthesis result
        for result in synthesis_results:
            if result.rendered_visual:
                # Simulated user feedback (in real usage, this would come from users)
                user_feedback = self._generate_simulated_feedback(result)

                # Perform evaluation
                evaluation = await self.evaluator.evaluate_synthesis(
                    result,
                    ground_truth=None,  # No ground truth for demo
                    user_feedback=user_feedback
                )

                evaluations.append(evaluation)

                logger.info(f"Evaluation for {result.synthesis_id}:")
                logger.info(f"  Overall score: {evaluation.overall_score:.2f}")
                logger.info(f"  Quality level: {evaluation.quality_level}")
                logger.info(f"  Recommendations: {len(evaluation.recommendations)}")

        # Perform adaptation based on evaluations
        if evaluations:
            adaptation_params = await self.evaluator.adapt_parameters(evaluations)

            logger.info("System adaptation performed:")
            logger.info(f"  Learning rate: {adaptation_params.learning_rate}")
            logger.info(f"  Feature extraction adjustments: {len(adaptation_params.feature_extraction_params)}")
            logger.info(f"  Perception adjustments: {len(adaptation_params.perception_params)}")
            logger.info(f"  Expression adjustments: {len(adaptation_params.expression_params)}")

        return evaluations

    async def demonstrate_real_time_potential(self):
        """Demonstrate real-time processing capabilities."""
        logger.info("Demonstrating real-time processing potential...")

        # Configure for real-time processing
        self.synthesizer.configure({
            'real_time_processing': True,
            'max_processing_latency': 50,  # 50ms target
            'quality_preset': 'balanced'
        })

        # Simulate real-time audio stream
        stream_duration = 5  # seconds
        sample_rate = 44100
        chunk_size = 2048

        logger.info(f"Simulating {stream_duration}s real-time audio stream...")

        start_time = asyncio.get_event_loop().time()
        chunks_processed = 0

        for i in range(0, int(stream_duration * sample_rate), chunk_size):
            # Generate audio chunk
            audio_chunk = self._generate_random_audio_chunk(chunk_size)

            # Process chunk
            result = await self.synthesizer.synthesize_audio_visual(audio_chunk, {
                'real_time': True,
                'chunk_id': chunks_processed
            })

            chunks_processed += 1

            # Check latency
            current_time = asyncio.get_event_loop().time()
            latency = (current_time - start_time) - (i / sample_rate)

            if latency > 0.1:  # More than 100ms latency
                logger.warning(f"High latency detected: {latency:.3f}s")

        total_time = asyncio.get_event_loop().time() - start_time
        logger.info(f"Real-time simulation complete: {chunks_processed} chunks in {total_time:.2f}s")
        logger.info(".2f")

    def _generate_birdsong_audio(self) -> np.ndarray:
        """Generate simulated birdsong audio."""
        sample_rate = 44100
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration))

        # Create melodic, harmonic birdsong-like sound
        frequency = 2000 + 500 * np.sin(2 * np.pi * 3 * t)  # Varying frequency
        audio = 0.3 * np.sin(2 * np.pi * frequency * t)

        # Add harmonics
        audio += 0.1 * np.sin(2 * np.pi * 2 * frequency * t)
        audio += 0.05 * np.sin(2 * np.pi * 3 * frequency * t)

        # Add some noise for texture
        audio += 0.02 * np.random.normal(0, 1, len(audio))

        return audio.astype(np.float32)

    def _generate_explosion_audio(self) -> np.ndarray:
        """Generate simulated explosion audio."""
        sample_rate = 44100
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))

        # Sharp attack, then decay
        envelope = np.exp(-3 * t)  # Exponential decay
        noise = np.random.normal(0, 1, len(t))

        # Low-frequency emphasis
        audio = envelope * noise

        # Add some rumble
        low_freq = 0.2 * np.sin(2 * np.pi * 50 * t) * envelope

        return (audio + low_freq).astype(np.float32)

    def _generate_alarm_audio(self) -> np.ndarray:
        """Generate simulated alarm audio."""
        sample_rate = 44100
        duration = 1.5
        t = np.linspace(0, duration, int(sample_rate * duration))

        # Alternating high-low tones
        freq1, freq2 = 800, 1200
        cycle_time = 0.3

        # Create alternating pattern
        cycle_phase = (t % cycle_time) / cycle_time
        frequency = np.where(cycle_phase < 0.5, freq1, freq2)

        audio = 0.4 * np.sin(2 * np.pi * frequency * t)

        return audio.astype(np.float32)

    def _generate_nature_audio(self) -> np.ndarray:
        """Generate simulated nature audio (wind/rain)."""
        sample_rate = 44100
        duration = 3.0
        samples = int(sample_rate * duration)

        # Wind-like noise with some tonal elements
        wind = 0.2 * np.random.normal(0, 1, samples)

        # Add some water droplet sounds
        t = np.linspace(0, duration, samples)
        droplets = np.zeros(samples)

        # Random droplet timings
        for _ in range(8):
            drop_time = np.random.uniform(0, duration)
            drop_duration = 0.05
            drop_mask = (t >= drop_time) & (t <= drop_time + drop_duration)
            drop_envelope = np.exp(-50 * (t[drop_mask] - drop_time))
            droplets[drop_mask] += 0.1 * np.random.normal(0, 1, np.sum(drop_mask)) * drop_envelope

        return (wind + droplets).astype(np.float32)

    def _generate_urban_audio(self) -> np.ndarray:
        """Generate simulated urban audio."""
        sample_rate = 44100
        duration = 2.0
        samples = int(sample_rate * duration)

        # Mix of traffic-like sounds and urban noise
        t = np.linspace(0, duration, samples)

        # Engine-like rumble
        engine = 0.15 * np.sin(2 * np.pi * 80 * t) * np.random.normal(0.8, 0.2, samples)

        # Horn sounds (occasional)
        horn = np.zeros(samples)
        for _ in range(3):
            horn_time = np.random.uniform(0.5, duration - 0.2)
            horn_freq = np.random.choice([400, 500, 600])
            horn_mask = (t >= horn_time) & (t <= horn_time + 0.15)
            horn_envelope = np.exp(-20 * (t[horn_mask] - horn_time))
            horn[horn_mask] += 0.2 * np.sin(2 * np.pi * horn_freq * t[horn_mask]) * horn_envelope

        # Background urban noise
        urban_noise = 0.1 * np.random.normal(0, 1, samples)

        return (engine + horn + urban_noise).astype(np.float32)

    def _generate_random_audio(self) -> np.ndarray:
        """Generate random test audio."""
        sample_rate = 44100
        duration = 1.0
        samples = int(sample_rate * duration)

        # Random harmonic sound
        t = np.linspace(0, duration, samples)
        base_freq = np.random.uniform(200, 2000)
        harmonics = np.random.randint(1, 4)

        audio = np.zeros(samples)
        for h in range(1, harmonics + 1):
            audio += (0.3 / h) * np.sin(2 * np.pi * base_freq * h * t)

        # Add noise
        audio += 0.1 * np.random.normal(0, 1, samples)

        return audio.astype(np.float32)

    def _generate_random_audio_chunk(self, chunk_size: int) -> np.ndarray:
        """Generate a small random audio chunk."""
        return 0.2 * np.random.normal(0, 1, chunk_size).astype(np.float32)

    def _generate_simulated_feedback(self, result) -> Dict[str, Any]:
        """Generate simulated user feedback for evaluation."""
        # Simulate realistic user feedback based on result quality
        base_satisfaction = 0.7

        if result.perception and result.expression:
            # Higher satisfaction for well-formed results
            satisfaction = base_satisfaction + np.random.uniform(-0.2, 0.3)
        else:
            satisfaction = base_satisfaction - np.random.uniform(0.1, 0.4)

        satisfaction = np.clip(satisfaction, 0, 1)

        feedback = {
            'satisfaction': satisfaction,
            'visual_quality': np.random.uniform(0.6, 1.0),
            'emotional_accuracy': np.random.uniform(0.5, 1.0),
            'creativity': np.random.uniform(0.6, 1.0),
            'appropriateness': np.random.uniform(0.7, 1.0),
            'comments': self._generate_feedback_comment(satisfaction)
        }

        return feedback

    def _generate_feedback_comment(self, satisfaction: float) -> str:
        """Generate a feedback comment based on satisfaction."""
        if satisfaction > 0.8:
            comments = [
                "Absolutely beautiful representation!",
                "This captures the essence perfectly.",
                "Very creative and emotionally resonant.",
                "Outstanding audio-visual synthesis!"
            ]
        elif satisfaction > 0.6:
            comments = [
                "Good representation of the sound.",
                "Interesting visual interpretation.",
                "Decent emotional capture.",
                "Solid audio-visual translation."
            ]
        else:
            comments = [
                "Could be more emotionally expressive.",
                "Visual representation feels disconnected.",
                "Needs more creativity.",
                "Doesn't quite capture the sound's essence."
            ]

        return np.random.choice(comments)

    async def run_full_demonstration(self):
        """Run the complete system demonstration."""
        logger.info("=== U-CogNet Audio-Visual Synthesis Demonstration ===")

        try:
            # Initialize system
            await self.initialize_system()

            # Demonstrate environmental sounds
            synthesis_results = await self.demonstrate_environmental_sounds()

            # Demonstrate batch processing
            await self.demonstrate_batch_processing()

            # Demonstrate evaluation and adaptation
            evaluations = await self.demonstrate_evaluation_and_adaptation(synthesis_results)

            # Demonstrate real-time potential
            await self.demonstrate_real_time_potential()

            # Show performance statistics
            perf_stats = self.synthesizer.get_performance_stats()
            logger.info("Final Performance Statistics:")
            logger.info(f"  Total syntheses: {perf_stats['total_syntheses']}")
            logger.info(f"  Average latency: {perf_stats['average_latency']:.2f}ms")
            logger.info(f"  Error count: {perf_stats['error_count']}")

            logger.info("=== Demonstration Complete ===")

        except Exception as e:
            logger.error(f"Demonstration failed: {e}")
            raise

        finally:
            # Cleanup
            await self.synthesizer.cleanup()
            await self.evaluator.cleanup()

async def main():
    """Main entry point for the demonstration."""
    demo = AudioVisualDemo()
    await demo.run_full_demonstration()

if __name__ == "__main__":
    # Run the demonstration
    asyncio.run(main())</content>
