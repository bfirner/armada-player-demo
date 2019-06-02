import random

class ArmadaDice:
    hit={ "red": 2, "blue": 4, "black": 4 }
    crit={ "red": 2, "blue": 2, "black": 0 }
    accuracy={ "red": 1, "blue": 2, "black": 0 }
    hit_hit={ "red": 1, "blue": 0, "black": 0 }
    hit_crit={ "red": 0, "blue": 0, "black": 2 }
    blank={ "red": 2, "blue": 0, "black": 2 }

    die_colors = ["red", "blue", "black"]
    die_faces = { "hit": hit,
                  "crit": crit,
                  "accuracy": accuracy,
                  "hit_hit": hit_hit,
                  "hit_crit": hit_crit,
                  "blank": blank}

    face_to_damage = { "hit": 1,
                       "crit": 1,
                       "accuracy": 0,
                       "hit_hit": 2,
                       "hit_crit": 2,
                       "blank": 0}

    @staticmethod
    def random_roll(color):
        rand_roll = random.uniform(0, 8)
        for face, counts in ArmadaDice.die_faces.items():
            if rand_roll <= counts[color]:
                return face
            else:
                rand_roll -= counts[color]

    @staticmethod
    def pool_damage(pool):
        """Calculate the damage in a dice pool.

        Args:
            pool (array of die_faces)
        Returns:
            Number of damage
        """
        return sum([ArmadaDice.face_to_damage[face] for face in pool])
