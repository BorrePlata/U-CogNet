#!/usr/bin/env python3
"""
U-CogNet Multimodal Attention Demo
Interactive demonstration of gating attention and hierarchical fusion.
Shows real-time attention modulation in a simplified Snake environment.
"""

import sys
import time
import pygame
import numpy as np
from pathlib import Path

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from snake_env import SnakeEnv
from snake_agent import IncrementalSnakeAgent
from snake_audio import SnakeAudioSystem
from multimodal_attention import (
    GatingAttentionController, Modality,
    create_visual_signal, create_audio_signal
)

class InteractiveAttentionDemo:
    """Interactive demo showing attention gating in real-time."""

    def __init__(self):
        self.width, self.height = 800, 600
        self.cell_size = 20
        self.grid_width, self.grid_height = 20, 20

        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("U-CogNet: Multimodal Attention Demo")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)

        # Initialize systems
        self.env = SnakeEnv(width=self.grid_width, height=self.grid_height)
        self.agent = IncrementalSnakeAgent()
        self.audio_system = SnakeAudioSystem(enabled=True)
        self.attention_controller = GatingAttentionController()

        # Demo state
        self.running = True
        self.paused = False
        self.episode = 0
        self.score = 0
        self.steps = 0
        self.show_attention_details = True

        # Colors for attention visualization
        self.colors = {
            'background': (20, 20, 30),
            'grid': (40, 40, 50),
            'snake_head': (100, 255, 100),
            'snake_body': (50, 200, 50),
            'food': (255, 100, 100),
            'text': (255, 255, 255),
            'attention_open': (100, 255, 100),
            'attention_closed': (255, 100, 100),
            'attention_filtering': (255, 255, 100),
            'panel_bg': (30, 30, 40),
            'panel_border': (100, 100, 120)
        }

    def run_demo(self):
        """Run the interactive attention demo."""
        print("🎮 U-CogNet Multimodal Attention Demo")
        print("Controls:")
        print("  SPACE: Pause/Resume")
        print("  R: Reset episode")
        print("  A: Toggle attention details")
        print("  Q: Quit")
        print("=" * 50)

        try:
            while self.running:
                self.handle_events()
                if not self.paused:
                    self.update()
                self.render()
                self.clock.tick(8)  # Slow enough to observe attention changes

        finally:
            self.audio_system.cleanup()
            pygame.quit()

    def handle_events(self):
        """Handle pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                    print(f"{'Paused' if self.paused else 'Resumed'}")
                elif event.key == pygame.K_r:
                    self.reset_episode()
                elif event.key == pygame.K_a:
                    self.show_attention_details = not self.show_attention_details

    def reset_episode(self):
        """Reset the current episode."""
        state = self.env.reset()
        self.episode += 1
        self.score = 0
        self.steps = 0
        print(f"Episode {self.episode} started")

    def update(self):
        """Update game state and attention system."""
        if self.steps == 0:
            self.reset_episode()

        # Get current state
        state = self.env._get_state()

        # Choose action
        action = self.agent.choose_action(state)

        # Execute action
        next_state, base_reward, done, _ = self.env.step(action)

        # Create multimodal signals
        game_state = {
            'snake': self.env.snake.copy(),
            'food': self.env.food,
            'score': self.env.score,
            'current_score': self.env.score,
            'episode_steps': self.steps + 1,
            'food_distance': abs(self.env.snake[0][0] - self.env.food[0]) + abs(self.env.snake[0][1] - self.env.food[1]),
            'danger_level': len(self.env.snake) / 400.0,
            'visual_clarity': 0.8,
            'audio_context': 'eating' if base_reward > 0 else 'dying' if base_reward < 0 else 'moving'
        }

        # Create modality signals
        modality_signals = []

        # Visual signal
        visual_data = {
            'snake_length': len(game_state['snake']),
            'food_distance': game_state['food_distance'],
            'danger_level': game_state['danger_level']
        }
        visual_signal = create_visual_signal(
            data=visual_data,
            confidence=min(0.9, 0.5 + game_state['visual_clarity']),
            priority=0.7
        )
        modality_signals.append(visual_signal)

        # Audio signal
        audio_data = {
            'reward_type': 'positive' if base_reward > 0 else 'negative',
            'intensity': abs(base_reward),
            'context': game_state['audio_context']
        }
        audio_signal = create_audio_signal(
            data=audio_data,
            confidence=0.8 if abs(base_reward) > 0 else 0.3,
            priority=0.6
        )
        modality_signals.append(audio_signal)

        # Process through attention system
        current_performance = game_state['current_score'] / max(1, game_state['episode_steps'])
        fused_signal, attention_state = self.attention_controller.process_multimodal_input(
            modality_signals, current_performance
        )

        # Calculate enhanced reward
        enhancement_factor = 1.0
        if fused_signal:
            enhancement_factor = 1.0 + (fused_signal.confidence * 0.1) + (fused_signal.priority * 0.05)

            # Modality-specific enhancements
            if (fused_signal.modality == Modality.VISUAL and
                attention_state.active_modalities[Modality.VISUAL].name == 'OPEN'):
                enhancement_factor *= 1.15
            elif (fused_signal.modality == Modality.AUDIO and
                  attention_state.active_modalities[Modality.AUDIO].name == 'OPEN'):
                enhancement_factor *= 1.10

            # Audio feedback only if gate is open
            if (fused_signal.modality == Modality.AUDIO and
                attention_state.active_modalities[Modality.AUDIO].name == 'OPEN'):
                if base_reward > 0:
                    self.audio_system.play_eat_sound()
                elif base_reward < 0:
                    self.audio_system.play_death_sound()

        reward = base_reward * enhancement_factor

        # Learn
        self.agent.learn(state, action, reward, next_state, done)

        # Update counters
        self.score = self.env.score
        self.steps += 1

        # Reset if done
        if done:
            time.sleep(1)  # Brief pause to observe final state
            self.reset_episode()

    def render(self):
        """Render the demo interface."""
        self.screen.fill(self.colors['background'])

        # Draw game area
        self.draw_game_area()

        # Draw attention panel
        if self.show_attention_details:
            self.draw_attention_panel()

        # Draw status bar
        self.draw_status_bar()

        pygame.display.flip()

    def draw_game_area(self):
        """Draw the Snake game area."""
        game_area_x = 50
        game_area_y = 50

        # Draw grid
        for x in range(self.grid_width):
            for y in range(self.grid_height):
                rect = pygame.Rect(
                    game_area_x + x * self.cell_size,
                    game_area_y + y * self.cell_size,
                    self.cell_size, self.cell_size
                )
                pygame.draw.rect(self.screen, self.colors['grid'], rect, 1)

        # Draw snake with attention-based coloring
        attention_status = self.attention_controller.get_attention_status()
        visual_gate = attention_status['active_gates'].get('visual', 'closed')
        audio_gate = attention_status['active_gates'].get('audio', 'closed')

        # Determine snake color based on attention
        if visual_gate == 'open' and audio_gate == 'open':
            head_color = (255, 255, 0)  # Yellow for multimodal
            body_color = (200, 200, 0)
        elif visual_gate == 'open':
            head_color = (0, 255, 255)  # Cyan for visual focus
            body_color = (0, 200, 200)
        elif audio_gate == 'open':
            head_color = (255, 0, 255)  # Magenta for audio focus
            body_color = (200, 0, 200)
        else:
            head_color = self.colors['snake_head']
            body_color = self.colors['snake_body']

        # Draw snake
        for i, segment in enumerate(self.env.snake):
            x, y = segment
            rect = pygame.Rect(
                game_area_x + x * self.cell_size,
                game_area_y + y * self.cell_size,
                self.cell_size, self.cell_size
            )
            color = head_color if i == 0 else body_color
            pygame.draw.rect(self.screen, color, rect)

        # Draw food
        food_x, food_y = self.env.food
        rect = pygame.Rect(
            game_area_x + food_x * self.cell_size,
            game_area_y + food_y * self.cell_size,
            self.cell_size, self.cell_size
        )
        pygame.draw.rect(self.screen, self.colors['food'], rect)

    def draw_attention_panel(self):
        """Draw the attention status panel."""
        panel_x = 500
        panel_y = 50
        panel_width = 280
        panel_height = 300

        # Panel background
        pygame.draw.rect(self.screen, self.colors['panel_bg'],
                        (panel_x, panel_y, panel_width, panel_height))
        pygame.draw.rect(self.screen, self.colors['panel_border'],
                        (panel_x, panel_y, panel_width, panel_height), 2)

        # Title
        title = self.font.render("Attention Gates", True, self.colors['text'])
        self.screen.blit(title, (panel_x + 10, panel_y + 10))

        # Get attention status
        attention_status = self.attention_controller.get_attention_status()

        y_offset = panel_y + 40

        # Visual gate
        visual_gate = attention_status['active_gates'].get('visual', 'closed')
        self.draw_gate_status("Visual", visual_gate, panel_x + 10, y_offset)
        y_offset += 30

        # Audio gate
        audio_gate = attention_status['active_gates'].get('audio', 'closed')
        self.draw_gate_status("Audio", audio_gate, panel_x + 10, y_offset)
        y_offset += 30

        # Text gate (placeholder)
        self.draw_gate_status("Text", "closed", panel_x + 10, y_offset)
        y_offset += 30

        # Tactile gate (placeholder)
        self.draw_gate_status("Tactile", "closed", panel_x + 10, y_offset)
        y_offset += 40

        # Performance metrics
        perf_text = ".2f"
        perf_surface = self.small_font.render(perf_text, True, self.colors['text'])
        self.screen.blit(perf_surface, (panel_x + 10, y_offset))

        y_offset += 20
        trend_text = ".2f"
        trend_surface = self.small_font.render(trend_text, True, self.colors['text'])
        self.screen.blit(trend_surface, (panel_x + 10, y_offset))

        y_offset += 30
        buffer_text = f"Buffer: {attention_status.get('temporal_buffer_size', 0)}"
        buffer_surface = self.small_font.render(buffer_text, True, self.colors['text'])
        self.screen.blit(buffer_surface, (panel_x + 10, y_offset))

    def draw_gate_status(self, name, status, x, y):
        """Draw a single gate status indicator."""
        # Gate name
        name_surface = self.small_font.render(f"{name}:", True, self.colors['text'])
        self.screen.blit(name_surface, (x, y))

        # Status indicator
        status_colors = {
            'open': self.colors['attention_open'],
            'closed': self.colors['attention_closed'],
            'filtering': self.colors['attention_filtering']
        }

        indicator_color = status_colors.get(status, self.colors['attention_closed'])
        pygame.draw.circle(self.screen, indicator_color, (x + 80, y + 8), 8)

        # Status text
        status_surface = self.small_font.render(status.upper(), True, self.colors['text'])
        self.screen.blit(status_surface, (x + 100, y))

    def draw_status_bar(self):
        """Draw the status bar at the bottom."""
        bar_y = self.height - 40

        # Background
        pygame.draw.rect(self.screen, self.colors['panel_bg'], (0, bar_y, self.width, 40))
        pygame.draw.line(self.screen, self.colors['panel_border'], (0, bar_y), (self.width, bar_y), 2)

        # Status texts
        texts = [
            f"Episode: {self.episode}",
            f"Score: {self.score}",
            f"Steps: {self.steps}",
            f"Q-States: {len(self.agent.q_table)}",
            "[SPACE] Pause  [R] Reset  [A] Toggle Details  [Q] Quit"
        ]

        x_offset = 10
        for text in texts:
            surface = self.small_font.render(text, True, self.colors['text'])
            self.screen.blit(surface, (x_offset, bar_y + 10))
            x_offset += 150

def main():
    """Run the interactive attention demo."""
    print("🎮 Starting U-CogNet Multimodal Attention Demo...")
    print("This demo shows how gating attention modulates multimodal learning in real-time.")
    print("Watch how the attention gates open/close based on learning performance!")
    print()

    demo = InteractiveAttentionDemo()
    demo.run_demo()

if __name__ == "__main__":
    main()