from __future__ import annotations

from .agents import NeedSeekingAgent
from .env import Action, HomeostaticSocialGrid


def main() -> None:
    env = HomeostaticSocialGrid(seed=7)
    agent = NeedSeekingAgent()
    obs = env.reset()

    print(env.render_ascii())
    print(_needs_line(obs))

    scripted_prelude = [Action.POINT, Action.ASK]
    for step_idx in range(20):
        action = (
            scripted_prelude[step_idx]
            if step_idx < len(scripted_prelude)
            else agent.act(obs)
        )
        obs, reward, terminated, truncated, info = env.step(action)
        print()
        print(f"action={action.value} reward={reward:+.3f} event={info['event']}")
        if obs.teacher_utterance:
            print(f'teacher="{obs.teacher_utterance}"')
        print(env.render_ascii())
        print(_needs_line(obs))
        if terminated or truncated:
            break


def _needs_line(obs) -> str:
    needs = obs.needs
    return (
        f"needs food={needs.food:.2f} water={needs.water:.2f} "
        f"energy={needs.energy:.2f} safety={needs.safety:.2f}"
    )


if __name__ == "__main__":
    main()
