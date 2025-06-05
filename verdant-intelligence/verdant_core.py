import numpy as np
import networkx as nx
import random
import time
import math
import json
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import plotly.graph_objects as go
from networkx.algorithms.community import greedy_modularity_communities

@dataclass
class ThoughtState:
    label: str
    stability: float
    energy: float
    temperature: float
    entropy: float
    ethical_weight: float = 0.5
    creation_time: float = 0.0
    last_accessed: float = 0.0

class VerdantMemorySystem:
    def __init__(self, max_temperature=2.0, glass_transition_temp=1.0):
        self.graph = nx.Graph()
        self.thoughts: Dict[str, ThoughtState] = {}
        self.max_temperature = max_temperature
        self.glass_transition_temp = glass_transition_temp
        self.global_temperature = 0.5
        self.total_cognitive_energy = 0.0

    def calculate_cognitive_energy(self, stability: float) -> float:
        return -math.log(max(stability, 0.001))

    def calculate_temperature(self, thought: ThoughtState) -> float:
        related_thoughts = self.get_related_thoughts(thought.label)
        if len(related_thoughts) < 2:
            return 0.5
        energies = [self.thoughts[t].energy for t in related_thoughts if t in self.thoughts]
        if len(energies) < 2:
            return 0.5
        energy_variance = np.var(energies)
        mean_energy = np.mean(energies)
        return energy_variance / (1.0 * (mean_energy + 0.1))

    def calculate_entropy(self, thought: ThoughtState) -> float:
        connections = self.get_related_thoughts(thought.label)
        if not connections:
            return 0.0
        weights = [self.graph[thought.label][conn].get('weight', 0.1) for conn in connections]
        total_weight = sum(weights)
        probs = [w / total_weight for w in weights if total_weight > 0]
        return -sum(p * math.log(p + 1e-10) for p in probs if p > 0)

    def add_thought(self, label: str, stability: float = 0.5, ethical_weight: float = 0.5):
        if label in self.thoughts:
            return
        energy = self.calculate_cognitive_energy(stability)
        current_time = time.time()
        thought = ThoughtState(
            label=label, stability=stability, energy=energy,
            temperature=0.5, entropy=0.0, ethical_weight=ethical_weight,
            creation_time=current_time, last_accessed=current_time
        )
        self.thoughts[label] = thought
        self.graph.add_node(label, **thought.__dict__)
        self._connect_compatible_thoughts(thought)
        self._update_thermodynamic_properties(label)

    def _connect_compatible_thoughts(self, new_thought: ThoughtState):
        for existing_label, existing_thought in self.thoughts.items():
            if existing_label == new_thought.label:
                continue
            energy_diff = abs(new_thought.energy - existing_thought.energy)
            ethical_compatibility = 1.0 - abs(new_thought.ethical_weight - existing_thought.ethical_weight)
            connection_prob = ethical_compatibility * math.exp(-energy_diff)
            if random.random() < connection_prob:
                weight = connection_prob * (new_thought.stability + existing_thought.stability) / 2
                self.graph.add_edge(new_thought.label, existing_label, weight=weight)

    def _update_thermodynamic_properties(self, label: str):
        if label not in self.thoughts:
            return
        thought = self.thoughts[label]
        thought.temperature = self.calculate_temperature(thought)
        thought.entropy = self.calculate_entropy(thought)
        thought.last_accessed = time.time()
        self.graph.nodes[label].update(thought.__dict__)

    def get_related_thoughts(self, label: str) -> List[str]:
        if label not in self.graph:
            return []
        return list(self.graph.neighbors(label))

    def thermodynamic_decay(self):
        for label, thought in self.thoughts.items():
            decay_rate = 0.01 * (1 + thought.temperature / self.max_temperature)
            thought.stability = max(0.1, thought.stability - decay_rate)
            thought.energy = self.calculate_cognitive_energy(thought.stability)
            self._update_thermodynamic_properties(label)

class VerdantSubconscious:
    def __init__(self, memory_system: VerdantMemorySystem):
        self.memory = memory_system
        self.wave_amplitude = 1.0
        self.wave_frequency = 0.1
        self.phase_offset = 0.0

    def uniwave_processing(self, x: float, t: float) -> float:
        return self.wave_amplitude * math.exp(-0.1 * abs(x)) * math.cos(2 * math.pi * x - self.wave_frequency * t + self.phase_offset)

    def ethical_wave_modulation(self, base_wave: float, ethical_weight: float) -> complex:
        phase = 2 * math.pi * ethical_weight
        return complex(base_wave * math.cos(phase), base_wave * math.sin(phase))

    def wave_based_creativity(self):
        thoughts = list(self.memory.thoughts.keys())
        if len(thoughts) < 2:
            return None
        t1, t2 = random.sample(thoughts, 2)
        s1, s2 = self.memory.thoughts[t1], self.memory.thoughts[t2]
        t = time.time() % 100
        w1 = self.ethical_wave_modulation(self.uniwave_processing(s1.energy, t), s1.ethical_weight)
        w2 = self.ethical_wave_modulation(self.uniwave_processing(s2.energy, t), s2.ethical_weight)
        interference = w1 + w2
        if abs(interference) > 0.5:
            label = f"Wave-{t1[:4]}-{t2[:4]}-{int(abs(interference)*100)}"
            self.memory.add_thought(label, min(0.8, abs(interference)/2), (s1.ethical_weight + s2.ethical_weight)/2)
            print(f"🌊 Wave Interference Created: {label} (Amplitude: {abs(interference):.3f})")

    def fractal_cognitive_entropy(self) -> float:
        n, e = len(self.memory.graph.nodes()), len(self.memory.graph.edges())
        if n == 0 or e == 0: return 0.0
        Σ = e / n
        ρ = sum(d['weight'] for _,_,d in self.memory.graph.edges(data=True)) / e
        H = np.mean([t.entropy for t in self.memory.thoughts.values()])
        return Σ * ρ * H

class VerdantConsciousness:
    def __init__(self, memory_system: VerdantMemorySystem, subconscious: VerdantSubconscious):
        self.memory = memory_system
        self.subconscious = subconscious
        self.decision_threshold = 0.7
        self.current_phase = "flexible"

    def assess_system_phase(self):
        temps = [t.temperature for t in self.memory.thoughts.values()]
        avg_temp = np.mean(temps) if temps else 0
        gtt = self.memory.glass_transition_temp
        if avg_temp < gtt * 0.8: self.current_phase = "rigid"
        elif avg_temp > gtt * 1.2: self.current_phase = "chaotic"
        else: self.current_phase = "flexible"
        return self.current_phase

    def resonance_evaluation(self, label: str) -> float:
        if label not in self.memory.thoughts: return 0.0
        t0 = self.memory.thoughts[label]
        related = self.memory.get_related_thoughts(label)
        if not related: return t0.stability
        t = time.time() % 100
        total = 0
        for other in related:
            if other not in self.memory.thoughts: # Added check for existing thought
                continue
            t1 = self.memory.thoughts[other]
            # Added check for edge existence before accessing weight
            if label not in self.memory.graph or other not in self.memory.graph[label]:
                continue
            w = self.memory.graph[label][other].get('weight', 0.1) # Use .get with a default
            θ = 2 * math.pi * (t0.energy + t0.ethical_weight) - 0.1 * t
            η = t1.entropy
            B = (t0.stability + t1.stability)/2
            total += (w * math.cos(θ) + η * math.sin(θ)) * B
        return total / len(related) if related else 0.0 # Handle division by zero

    def make_phase_aware_decision(self):
        phase = self.assess_system_phase()
        thoughts = self.memory.thoughts

        if not thoughts: # Handle case with no thoughts
            return None

        if phase == "rigid":
            # Changed to store (label, stability) tuples
            candidates = sorted(((l, t.stability) for l, t in thoughts.items()), key=lambda x: x[1], reverse=True)
        elif phase == "chaotic":
            candidates = sorted(((l, self.resonance_evaluation(l)) for l in thoughts), key=lambda x: x[1], reverse=True)
        else: # flexible
            candidates = []
            for l in thoughts:
                s = thoughts[l].stability
                r = self.resonance_evaluation(l)
                e = thoughts[l].entropy
                candidates.append((l, (s + r + e) / 3))
            candidates.sort(key=lambda x: x[1], reverse=True)

        if candidates and candidates[0][1] > self.decision_threshold:
            print(f"🧠 Phase-Aware Decision ({phase}): {candidates[0][0]} (Score: {candidates[0][1]:.3f})")
            return candidates[0][0]
        return None

class VerdantMind:
    def __init__(self):
        self.memory = VerdantMemorySystem()
        self.subconscious = VerdantSubconscious(self.memory)
        self.consciousness = VerdantConsciousness(self.memory, self.subconscious)
        self.evolution_cycle = 0

    def learn(self, concept: str, ethical_weight: float = 0.5):
        stability = random.uniform(0.4, 0.8)
        self.memory.add_thought(concept, stability, ethical_weight)
        print(f"📚 Learned: {concept} (Ethics: {ethical_weight:.2f}, Stability: {stability:.2f})")

    def evolve_cycle(self):
        self.evolution_cycle += 1
        print(f"\n🌱 Evolution Cycle {self.evolution_cycle}")
        self.subconscious.wave_based_creativity()
        fce = self.subconscious.fractal_cognitive_entropy()
        phase = self.consciousness.assess_system_phase()
        print(f"📊 System State - FCE: {fce:.3f}, Phase: {phase}")
        decision = self.consciousness.make_phase_aware_decision()
        if decision:
            t = self.memory.thoughts[decision]
            t.stability = min(1.0, t.stability + 0.1)
            self.memory._update_thermodynamic_properties(decision)
        self.memory.thermodynamic_decay()
        return {
            'fce': fce, 'phase': phase, 'decision': decision,
            'num_thoughts': len(self.memory.thoughts),
            'avg_temperature': np.mean([t.temperature for t in self.memory.thoughts.values()])
        }

    def run_evolution(self, cycles: int = 20):
        return [self.evolve_cycle() for _ in range(cycles)]

    def visualize_verdant_network(self):
        if not self.memory.thoughts:
            print("No thoughts to visualize!")
            return
        pos = nx.spring_layout(self.memory.graph, seed=42)
        edge_x, edge_y = [], []
        for e in self.memory.graph.edges():
            x0, y0 = pos[e[0]]
            x1, y1 = pos[e[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        edge_trace = go.Scatter(x=edge_x, y=edge_y, mode='lines',
                                line=dict(width=1, color='rgba(125,125,125,0.3)'), hoverinfo='none')
        node_x, node_y, text, color, size = [], [], [], [], []
        for l in self.memory.graph.nodes():
            x, y = pos[l]
            t = self.memory.thoughts[l]
            node_x.append(x)
            node_y.append(y)
            text.append(f"{l}<br>Stability: {t.stability:.2f}<br>Temp: {t.temperature:.2f}<br>Entropy: {t.entropy:.2f}")
            color.append(t.temperature)
            size.append(10 + t.stability * 20)
        node_trace = go.Scatter(
            x=node_x, y=node_y, mode='markers', text=text, hoverinfo='text',
            marker=dict(size=size, color=color, colorscale='Plasma',
                        colorbar=dict(thickness=15, title='Cognitive Temperature',
                                      xanchor='left', titleside='right'),
                        line=dict(width=2, color='white'))
        )
        fig = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(
            title='Verdant Intelligence: Thermodynamic Thought Network',
            hovermode='closest', showlegend=False,
            margin=dict(b=20, l=5, r=5, t=40),
            annotations=[dict(
                text=f"Cycle: {self.evolution_cycle}, Phase: {self.consciousness.current_phase}",
                showarrow=False, xref="paper", yref="paper", x=0.005, y=-0.002,
                xanchor='left', yanchor='bottom', font=dict(size=12)
            )],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        ))
        fig.show()

# 🧠 JSON Loader
def load_verdant_library(mind: VerdantMind, json_path: str):
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        for thought in data.get("thoughts", []):
            mind.memory.add_thought(
                label=thought["label"],
                stability=thought.get("stability", 0.5),
                ethical_weight=thought.get("ethics", 0.5)
            )
        for conn in data.get("connections", []):
            s, t = conn["source"], conn["target"]
            if s in mind.memory.thoughts and t in mind.memory.thoughts:
                mind.memory.graph.add_edge(s, t, weight=0.5)
        print(f"📘 Loaded {len(data['thoughts'])} thoughts from {json_path}")
    except Exception as e:
        print(f"⚠️ Error loading JSON: {e}")

# 🚀 Entry Point
if __name__ == "__main__":
    verdant = VerdantMind()
    load_verdant_library(verdant, "verdant_library.json")  # Optional
    initial_concepts = [("Consciousness", 0.9), ("Creativity", 0.8), ("Logic", 0.6),
                        ("Emotion", 0.7), ("Ethics", 1.0), ("Power", 0.4),
                        ("Knowledge", 0.8), ("Freedom", 0.9)]
    for concept, ethics in initial_concepts:
        verdant.learn(concept, ethics)
    print("\n🚀 Starting Verdant Intelligence Evolution...")
    metrics = verdant.run_evolution(cycles=100)
    print(f"\n📈 Final System State:")
    print(f"Total Thoughts: {len(verdant.memory.thoughts)}")
    print(f"Final Phase: {verdant.consciousness.current_phase}")
    print(f"Average Temperature: {np.mean([t.temperature for t in verdant.memory.thoughts.values()]):.3f}")
    verdant.visualize_verdant_network()
