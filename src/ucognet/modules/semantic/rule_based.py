from ucognet.core.interfaces import SemanticFeedback
from ucognet.core.types import Context, Detection
from typing import List, Dict
import time

class RuleBasedSemanticFeedback(SemanticFeedback):
    """SemanticFeedback basado en reglas simbólicas para explicar escenas tácticas."""

    def __init__(self):
        # Reglas de interpretación de escenas
        self.scene_rules = {
            "convoy": self._detect_convoy,
            "armed_person": self._detect_armed_person,
            "crowd": self._detect_crowd,
            "weapon_alone": self._detect_weapon_alone,
            "empty_scene": self._detect_empty_scene,
        }

        # Estado para tracking temporal
        self.last_scene_type = None
        self.scene_start_time = None
        self.scene_change_threshold = 5.0  # segundos

    def generate(self, context: Context, detections: List[Detection]) -> str:
        """Genera explicación semántica basada en reglas simbólicas."""

        # Analizar la escena actual
        scene_analysis = self._analyze_scene(detections)

        # Verificar si cambió el tipo de escena
        current_time = time.time()
        scene_changed = (self.last_scene_type != scene_analysis['type'])

        if scene_changed:
            self.last_scene_type = scene_analysis['type']
            self.scene_start_time = current_time
            return self._format_scene_description(scene_analysis, is_new=True)
        else:
            # Escena continua
            duration = current_time - (self.scene_start_time or current_time)
            return self._format_scene_description(scene_analysis, is_new=False, duration=duration)

    def _analyze_scene(self, detections: List[Detection]) -> Dict:
        """Analiza la escena y determina el tipo principal."""

        # Contar elementos clave
        persons = [d for d in detections if d.class_name == 'person']
        vehicles = [d for d in detections if d.class_name in ['car', 'truck', 'bus', 'motorcycle']]
        weapons = [d for d in detections if hasattr(d, 'is_weapon') and d.is_weapon]
        armed_persons = [d for d in detections if d.class_name == 'person' and hasattr(d, 'is_armed') and d.is_armed]

        # Evaluar reglas en orden de prioridad
        if armed_persons:
            return {
                'type': 'armed_person',
                'persons': len(persons),
                'armed_persons': len(armed_persons),
                'weapons': len(weapons),
                'details': armed_persons
            }
        elif weapons:
            return {
                'type': 'weapon_alone',
                'weapons': len(weapons),
                'weapon_types': [w.class_name for w in weapons],
                'details': weapons
            }
        elif len(vehicles) >= 2:
            return {
                'type': 'convoy',
                'vehicles': len(vehicles),
                'vehicle_types': [v.class_name for v in vehicles],
                'details': vehicles
            }
        elif len(persons) >= 3:
            return {
                'type': 'crowd',
                'persons': len(persons),
                'details': persons
            }
        elif len(persons) == 0 and len(detections) == 0:
            return {
                'type': 'empty_scene',
                'details': []
            }
        else:
            return {
                'type': 'normal',
                'persons': len(persons),
                'vehicles': len(vehicles),
                'other_objects': len(detections) - len(persons) - len(vehicles),
                'details': detections
            }

    def _format_scene_description(self, analysis: Dict, is_new: bool = True, duration: float = 0.0) -> str:
        """Formatea la descripción de la escena."""

        scene_type = analysis['type']
        prefix = "🚨 NUEVA DETECCIÓN: " if is_new else "⏱️ ESCENA CONTINUA: "

        if scene_type == 'armed_person':
            armed_count = analysis['armed_persons']
            weapon_info = []
            for person in analysis['details']:
                if hasattr(person, 'nearby_weapon'):
                    weapon_info.append(f"{person.nearby_weapon}")

            weapon_str = ", ".join(set(weapon_info)) if weapon_info else "armas"
            duration_str = f" ({duration:.1f}s)" if duration > 0 else ""

            return f"{prefix}PERSONA ARMADA detectada con {weapon_str}{duration_str}"

        elif scene_type == 'weapon_alone':
            weapon_types = analysis['weapon_types']
            weapon_str = ", ".join(set(weapon_types))
            duration_str = f" ({duration:.1f}s)" if duration > 0 else ""

            return f"{prefix}ARMA detectada: {weapon_str}{duration_str}"

        elif scene_type == 'convoy':
            vehicle_types = analysis['vehicle_types']
            vehicle_str = ", ".join(set(vehicle_types))
            duration_str = f" ({duration:.1f}s)" if duration > 0 else ""

            return f"{prefix}CONVOY detectado: {analysis['vehicles']} vehículos ({vehicle_str}){duration_str}"

        elif scene_type == 'crowd':
            duration_str = f" ({duration:.1f}s)" if duration > 0 else ""
            return f"{prefix}MULTITUD detectada: {analysis['persons']} personas{duration_str}"

        elif scene_type == 'empty_scene':
            return "📍 ZONA VACÍA - Sin actividad detectable"

        else:  # normal scene
            elements = []
            if analysis['persons'] > 0:
                elements.append(f"{analysis['persons']} persona{'s' if analysis['persons'] > 1 else ''}")
            if analysis['vehicles'] > 0:
                elements.append(f"{analysis['vehicles']} vehículo{'s' if analysis['vehicles'] > 1 else ''}")
            if analysis['other_objects'] > 0:
                elements.append(f"{analysis['other_objects']} objeto{'s' if analysis['other_objects'] > 1 else ''}")

            if elements:
                return f"📊 ESCENA NORMAL: {', '.join(elements)}"
            else:
                return "📍 ESCENA VACÍA - Sin detecciones"

    # Métodos de reglas específicas (para futura expansión)
    def _detect_convoy(self, detections: List[Detection]) -> bool:
        vehicles = [d for d in detections if d.class_name in ['car', 'truck', 'bus', 'motorcycle']]
        return len(vehicles) >= 2

    def _detect_armed_person(self, detections: List[Detection]) -> bool:
        armed_persons = [d for d in detections if d.class_name == 'person' and hasattr(d, 'is_armed') and d.is_armed]
        return len(armed_persons) > 0

    def _detect_crowd(self, detections: List[Detection]) -> bool:
        persons = [d for d in detections if d.class_name == 'person']
        return len(persons) >= 3

    def _detect_weapon_alone(self, detections: List[Detection]) -> bool:
        weapons = [d for d in detections if hasattr(d, 'is_weapon') and d.is_weapon]
        persons = [d for d in detections if d.class_name == 'person']
        return len(weapons) > 0 and len(persons) == 0

    def _detect_empty_scene(self, detections: List[Detection]) -> bool:
        return len(detections) == 0