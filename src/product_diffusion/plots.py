"""Figures for the product-diffusion case.

Colors are shared across the note so a crimson mark always means the minimum
hiring plan and a blue mark means a plan that still reaches everyone.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.ticker import MaxNLocator

INK = "#1f2430"
MUTED = "#667085"
RULE = "#d0d5dd"
CRIMSON = "#9b2335"
BLUE = "#1f4e79"
TEAL = "#1f6f6a"
SAND = "#f4f1ea"
PALE = "#e7eef6"
INFEASIBLE = "#c5cad3"


def render(payload: dict, directory: str | Path) -> list[Path]:
    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    _style()
    written = [
        _network(payload, output / "01_network.png"),
        _coverage(payload, output / "02_coverage.png"),
        _landscape(payload, output / "03_cost_landscape.png"),
        _penalty(payload, output / "04_penalty.png"),
        _hamiltonian(payload, output / "05_hamiltonian.png"),
        _circuit(payload, output / "06_qaoa_layer.png"),
        _search(payload, output / "07_parameter_search.png"),
        _depth(payload, output / "08_depth.png"),
        _measurement(payload, output / "09_measurement.png"),
        _solution(payload, output / "10_solution.png"),
    ]
    plt.close("all")
    return written


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "mathtext.fontset": "dejavusans",
        }
    )


def _save(figure: plt.Figure, path: Path) -> Path:
    figure.savefig(path, dpi=180, bbox_inches="tight", pad_inches=0.25)
    plt.close(figure)
    return path


def _advertising_positions() -> dict[str, tuple[float, float]]:
    return {
        "Enrique": (-2.15, 0.05),
        "Sarah": (-0.85, 1.2),
        "Raúl": (-0.85, -1.1),
        "Mario": (0.45, 0.05),
        "Ana": (1.85, 0.05),
        "Saúl": (3.2, 0.05),
    }


def _draw_graph(
    ax: plt.Axes,
    names: list[str],
    edges: list[list[str]],
    positions: dict[str, tuple[float, float]],
    *,
    node_face: dict[str, str],
    edge_color: dict[tuple[str, str], str] | None = None,
    edge_width: dict[tuple[str, str], float] | None = None,
    node_edge: dict[str, str] | None = None,
) -> None:
    edge_color = edge_color or {}
    edge_width = edge_width or {}
    node_edge = node_edge or {}
    for person, friend in edges:
        key = _edge_key(person, friend)
        start = positions[person]
        end = positions[friend]
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            color=edge_color.get(key, "#8b93a1"),
            lw=edge_width.get(key, 1.8),
            solid_capstyle="round",
            zorder=1,
        )
    for name in names:
        x, y = positions[name]
        ax.scatter(
            [x],
            [y],
            s=560,
            c=node_face.get(name, "#f7f5f2"),
            edgecolors=node_edge.get(name, INK),
            linewidths=1.3,
            zorder=3,
        )
        ax.text(x, y - 0.34, name, ha="center", va="top", fontsize=11, color=INK, zorder=4)
    ax.set_aspect("equal")
    ax.axis("off")


def _edge_key(person: str, friend: str) -> tuple[str, str]:
    return tuple(sorted((person, friend)))


def _network(payload: dict, path: Path) -> Path:
    figure, ax = plt.subplots(figsize=(7.4, 4.6))
    names = payload["names"]
    face = {name: "#f7f5f2" for name in names}
    _draw_graph(ax, names, payload["edges"], _advertising_positions(), node_face=face)
    ax.set_title("Friendships along which the product can travel", loc="left")
    _pad_graph(ax)
    return _save(figure, path)


def _pad_graph(ax: plt.Axes) -> None:
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    ax.set_xlim(x0 - 0.15, x1 + 0.15)
    ax.set_ylim(y0 - 0.35, y1 + 0.25)


def _coverage(payload: dict, path: Path) -> Path:
    names = payload["names"]
    neighborhoods = payload["closed_neighborhoods"]
    matrix = np.zeros((len(names), len(names)))
    for row, name in enumerate(names):
        for member in neighborhoods[name]:
            matrix[row, names.index(member)] = 1.0

    figure, ax = plt.subplots(figsize=(6.6, 5.4))
    ax.imshow(matrix, cmap=plt.matplotlib.colors.ListedColormap([SAND, BLUE]), vmin=0, vmax=1)
    ax.set_xticks(range(len(names)), names, rotation=0)
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("Person who hears the product")
    ax.set_ylabel("Person hired")
    ax.set_title("Who is reached by hiring one person", loc="left")
    ax.set_xticks(np.arange(len(names) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(names) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", lw=3)
    ax.tick_params(which="minor", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    figure.tight_layout()
    return _save(figure, path)


def _landscape(payload: dict, path: Path) -> Path:
    plans = payload["plans"]
    costs = np.array([plan["cost"] for plan in plans])
    order = np.argsort(costs, kind="mergesort")
    ordered = [plans[index] for index in order]
    ordered_costs = costs[order]

    figure, axes = plt.subplots(1, 2, figsize=(10.6, 4.5), gridspec_kw={"width_ratios": [1.35, 1]})
    axes[0].bar(
        np.arange(len(ordered)),
        ordered_costs,
        color=[_plan_color(plan) for plan in ordered],
        width=1.0,
        linewidth=0,
    )
    axes[0].set_xlabel("Hiring plans, ordered by cost")
    axes[0].set_ylabel("Cost")
    axes[0].set_xlim(-0.8, len(ordered) - 0.2)
    axes[0].set_title("All 64 plans", loc="left")
    optimum_rank = next(index for index, plan in enumerate(ordered) if plan["optimal"])
    axes[0].annotate(
        "Ana, Enrique",
        xy=(optimum_rank, payload["optimal_cost"]),
        xytext=(7, 7.2),
        color=CRIMSON,
        fontsize=10,
        arrowprops={"arrowstyle": "->", "color": CRIMSON, "lw": 0.9},
    )
    optimal_counts, dominating_counts, infeasible_counts = [], [], []
    cost_values = sorted({round(plan["cost"], 8) for plan in plans})
    for value in cost_values:
        selected = [plan for plan in plans if abs(plan["cost"] - value) < 1e-8]
        optimal_counts.append(sum(plan["optimal"] for plan in selected))
        dominating_counts.append(sum(plan["dominating"] and not plan["optimal"] for plan in selected))
        infeasible_counts.append(sum(not plan["dominating"] for plan in selected))
    positions = np.arange(len(cost_values))
    axes[1].bar(positions, infeasible_counts, color=INFEASIBLE, width=0.72)
    axes[1].bar(positions, dominating_counts, bottom=infeasible_counts, color=BLUE, width=0.72)
    bottom = np.array(infeasible_counts) + np.array(dominating_counts)
    axes[1].bar(positions, optimal_counts, bottom=bottom, color=CRIMSON, width=0.72)
    axes[1].set_xticks(positions, [_tick(value) for value in cost_values])
    axes[1].set_xlabel("Cost")
    axes[1].set_ylabel("Number of plans")
    axes[1].set_title("Plans at each cost", loc="left")
    figure.legend(
        handles=_plan_handles(),
        frameon=False,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.0),
    )
    figure.tight_layout(rect=(0, 0.08, 1, 1))
    return _save(figure, path)


def _plan_color(plan: dict) -> str:
    if plan["optimal"]:
        return CRIMSON
    if plan["dominating"]:
        return BLUE
    return INFEASIBLE


def _plan_handles() -> list[Rectangle]:
    return [
        Rectangle((0, 0), 1, 1, color=CRIMSON, label="Minimum plan"),
        Rectangle((0, 0), 1, 1, color=BLUE, label="Reaches everyone, costs more"),
        Rectangle((0, 0), 1, 1, color=INFEASIBLE, label="Leaves someone out"),
    ]


def _tick(value: float) -> str:
    if abs(value - round(value)) < 1e-8:
        return str(int(round(value)))
    return f"{value:g}"


def _penalty(payload: dict, path: Path) -> Path:
    sweep = payload["penalty_sweep"]
    penalties = np.array([row["penalty"] for row in sweep])
    dominating = np.array([row["best_dominating_cost"] for row in sweep])
    infeasible = np.array([row["best_infeasible_cost"] for row in sweep])

    figure = plt.figure(figsize=(11.2, 4.5))
    grid = figure.add_gridspec(1, 3, width_ratios=[1.45, 1, 1], wspace=0.28)
    ax = figure.add_subplot(grid[0, 0])
    ax.plot(penalties, dominating, color=BLUE, lw=2.0, label="Best plan that reaches everyone")
    ax.plot(penalties, infeasible, color=CRIMSON, lw=2.0, label="Best plan that leaves someone out")
    ymax = max(float(dominating.max()), float(infeasible.max())) * 1.08
    ax.set_xlim(0, float(penalties[-1]))
    ax.set_ylim(0, ymax)
    ax.axvline(0.5, color=MUTED, lw=0.8, ls=":")
    ax.axvline(payload["penalty"], color=INK, lw=0.9, ls="--")
    ax.text(0.62, 3.15, "B = 1/2", color=MUTED, fontsize=10, ha="left", va="bottom")
    ax.text(2.08, 2.55, "B = 2", color=INK, fontsize=10, ha="left", va="bottom")
    ax.set_xlabel("Penalty B on each unreached person")
    ax.set_ylabel("Cost")
    ax.set_title("Advertising network", loc="left")
    ax.legend(frameon=False, loc="upper left", fontsize=9)

    _cycle_panel(
        figure.add_subplot(grid[0, 1]),
        payload["cycle"],
        payload["cycle"]["squared_ground"][0],
        "Squared penalty",
    )
    _cycle_panel(
        figure.add_subplot(grid[0, 2]),
        payload["cycle"],
        payload["cycle"]["product_ground"][0],
        "Product penalty",
    )
    return _save(figure, path)


def _cycle_panel(ax: plt.Axes, cycle: dict, plan: dict, title: str) -> None:
    names = cycle["names"]
    positions = {"0": (0.0, 1.0), "1": (1.15, 1.0), "2": (1.15, -0.15), "3": (0.0, -0.15)}
    hired = set(plan["hired"])
    reached = _reached(names, cycle["edges"], hired)
    missed = [name for name in names if name not in reached]
    face = {}
    border = {}
    for name in names:
        if name in hired:
            face[name] = INK
            border[name] = INK
        elif name in missed:
            face[name] = "white"
            border[name] = CRIMSON
        else:
            face[name] = PALE
            border[name] = BLUE
    _draw_graph(
        ax,
        names,
        cycle["edges"],
        positions,
        node_face=face,
        node_edge=border,
    )
    ax.set_title(title, loc="left", fontsize=11)
    hired_label = ", ".join(plan["hired"]) if plan["hired"] else "nobody"
    if missed:
        status = f"hires {hired_label};  {', '.join(missed)} not reached"
    else:
        status = f"hires {hired_label};  everyone reached"
    ax.text(0.57, -0.95, status, ha="center", va="top", fontsize=8.5, color=INK)
    ax.set_xlim(-0.55, 1.75)
    ax.set_ylim(-1.55, 1.6)


def _reached(names: list[str], edges: list[list[str]], hired: set[str]) -> set[str]:
    adjacent: dict[str, set[str]] = {name: set() for name in names}
    for person, friend in edges:
        adjacent[person].add(friend)
        adjacent[friend].add(person)
    return {name for name in names if name in hired or adjacent[name] & hired}


def _hamiltonian(payload: dict, path: Path) -> Path:
    terms = payload["pauli_terms"]
    weights = sorted({int(term["weight"]) for term in terms})
    palette = {1: "#8aa0b8", 2: BLUE, 3: TEAL, 4: CRIMSON}
    figure, ax = plt.subplots(figsize=(7.6, 4.4))
    for index, term in enumerate(terms):
        weight = int(term["weight"])
        ax.plot([index, index], [0, term["coefficient"]], color=palette.get(weight, INK), lw=2.0)
        ax.scatter([index], [term["coefficient"]], s=18, color=palette.get(weight, INK), zorder=3)
    ax.axhline(0, color=RULE, lw=0.8)
    ax.set_xlabel("Pauli term, grouped by weight")
    ax.set_ylabel("Coefficient")
    ax.set_title("Cost operator, identity omitted", loc="left")
    handles = [
        Line2D([0], [0], color=palette[weight], marker="o", lw=2, label=f"weight {weight}")
        for weight in weights
    ]
    ax.legend(handles=handles, frameon=False, ncol=len(weights), loc="upper right")
    figure.tight_layout()
    return _save(figure, path)


def _circuit(payload: dict, path: Path) -> Path:
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.6), gridspec_kw={"width_ratios": [1.35, 1]})
    _schematic(axes[0], payload["names"])
    depths = payload["depths"]
    gates = ["h", "rx", "rz", "cx"]
    labels = {"h": "H", "rx": r"$R_X$", "rz": r"$R_Z$", "cx": "CX"}
    colors = {"h": "#8aa0b8", "rx": TEAL, "rz": BLUE, "cx": CRIMSON}
    x = np.arange(len(depths))
    width = 0.18
    for offset, gate in enumerate(gates):
        counts = [depth["gate_counts"].get(gate, 0) for depth in depths]
        axes[1].bar(x + (offset - 1.5) * width, counts, width=width, color=colors[gate], label=labels[gate])
    axes[1].set_xticks(x, [f"p = {depth['depth']}" for depth in depths])
    axes[1].set_ylabel("Gate count")
    axes[1].set_title("After translation to H, $R_X$, $R_Z$, CX", loc="left")
    axes[1].legend(frameon=False, ncol=4, loc="upper left")
    figure.tight_layout()
    return _save(figure, path)


def _schematic(ax: plt.Axes, names: list[str]) -> None:
    n = len(names)
    ys = np.linspace(n - 1, 0, n)
    for y in ys:
        ax.plot([0.2, 11.4], [y, y], color=INK, lw=1.0, zorder=1)
    for name, y in zip(names, ys):
        ax.text(-0.05, y, name, ha="right", va="center", fontsize=10)
        _gate_box(ax, 1.15, y, 0.62, 0.46, "H")
        _gate_box(ax, 9.15, y, 1.35, 0.46, r"$R_X(2\beta)$")
    box = FancyBboxPatch(
        (2.55, -0.42),
        4.7,
        n - 1 + 0.84,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        facecolor=PALE,
        edgecolor=INK,
        lw=1.1,
        zorder=2,
    )
    ax.add_patch(box)
    ax.text(4.9, (n - 1) / 2, r"$e^{-i\gamma H_C}$", ha="center", va="center", fontsize=15, zorder=3)
    ax.set_xlim(-1.7, 12.0)
    ax.set_ylim(-0.85, n - 0.15)
    ax.axis("off")
    ax.set_title("One layer", loc="left")


def _gate_box(ax: plt.Axes, x: float, y: float, width: float, height: float, label: str) -> None:
    patch = FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        facecolor="white",
        edgecolor=INK,
        lw=1.0,
        zorder=3,
    )
    ax.add_patch(patch)
    ax.text(x, y, label, ha="center", va="center", fontsize=9, zorder=4)


def _search(payload: dict, path: Path) -> Path:
    final = _operating_depth(payload)
    figure, axes = plt.subplots(1, 2, figsize=(10.4, 4.3))
    trace = final["best_trace"]
    axes[0].plot(np.arange(1, len(trace) + 1), trace, color=BLUE, lw=1.6)
    axes[0].axhline(payload["optimal_cost"], color=CRIMSON, lw=1.0, ls="--")
    axes[0].set_xlabel("Cost evaluations of the best restart")
    axes[0].set_ylabel(r"Expectation of $H_C$")
    axes[0].set_title(f"Best of {payload['restarts']} restarts, p = {final['depth']}", loc="left")

    finals = sorted(final["final_energies"])
    axes[1].plot(np.arange(1, len(finals) + 1), finals, "o", color=BLUE, ms=5)
    axes[1].axhline(payload["optimal_cost"], color=CRIMSON, lw=1.0, ls="--", label="Certified minimum")
    axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[1].set_xlabel("Restart, ordered by final expectation")
    axes[1].set_ylabel(r"Final expectation of $H_C$")
    axes[1].set_title("Where the restarts stop", loc="left")
    axes[1].legend(frameon=False, loc="best")
    figure.tight_layout()
    return _save(figure, path)


def _depth(payload: dict, path: Path) -> Path:
    depths = payload["depths"]
    labels = [str(record["depth"]) for record in depths]
    x = np.arange(len(depths))
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))
    axes[0].bar(x, [record["optimal_probability"] for record in depths], color=CRIMSON, width=0.62)
    axes[0].set_xticks(x, labels)
    axes[0].set_ylim(0, 1)
    axes[0].set_xlabel("Layers p")
    axes[0].set_ylabel("Probability of the minimum plan")
    axes[0].set_title("Mass on Ana and Enrique", loc="left")
    for index, record in enumerate(depths):
        axes[0].text(
            index,
            record["optimal_probability"] + 0.03,
            f"{record['optimal_probability']:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    energies = [record["best_energy"] for record in depths]
    axes[1].plot(x, energies, "o-", color=BLUE, lw=1.8, ms=7, label="Best expectation")
    axes[1].axhline(payload["optimal_cost"], color=CRIMSON, lw=1.0, ls="--", label="Certified minimum")
    axes[1].set_xticks(x, labels)
    axes[1].set_xlabel("Layers p")
    axes[1].set_ylabel(r"Expectation of $H_C$")
    axes[1].set_title("Energy after the same search budget", loc="left")
    axes[1].legend(frameon=False, loc="best")
    figure.tight_layout()
    return _save(figure, path)


def _measurement(payload: dict, path: Path) -> Path:
    final = _operating_depth(payload)
    plans = payload["plans"]
    paired = list(zip(plans, final["probabilities"], strict=True))
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.8), gridspec_kw={"width_ratios": [1, 1.25]})

    cost_values = sorted({round(plan["cost"], 8) for plan, _probability in paired})
    optimal_mass, dominating_mass, infeasible_mass = [], [], []
    for value in cost_values:
        optimal_mass.append(
            sum(probability for plan, probability in paired if plan["optimal"] and abs(plan["cost"] - value) < 1e-8)
        )
        dominating_mass.append(
            sum(
                probability
                for plan, probability in paired
                if plan["dominating"] and not plan["optimal"] and abs(plan["cost"] - value) < 1e-8
            )
        )
        infeasible_mass.append(
            sum(probability for plan, probability in paired if not plan["dominating"] and abs(plan["cost"] - value) < 1e-8)
        )
    keep = [
        index
        for index, value in enumerate(cost_values)
        if optimal_mass[index] + dominating_mass[index] + infeasible_mass[index] >= 0.01
    ]
    positions = np.arange(len(keep))
    axes[0].bar(positions, [infeasible_mass[index] for index in keep], color=INFEASIBLE, width=0.72)
    axes[0].bar(
        positions,
        [dominating_mass[index] for index in keep],
        bottom=[infeasible_mass[index] for index in keep],
        color=BLUE,
        width=0.72,
    )
    bottom = [
        infeasible_mass[index] + dominating_mass[index]
        for index in keep
    ]
    axes[0].bar(positions, [optimal_mass[index] for index in keep], bottom=bottom, color=CRIMSON, width=0.72)
    axes[0].set_xticks(positions, [_tick(cost_values[index]) for index in keep])
    axes[0].set_xlabel("Cost")
    axes[0].set_ylabel("Probability")
    axes[0].set_title("Probability by cost", loc="left")

    ranked = sorted(paired, key=lambda item: item[1], reverse=True)[:8]
    ranked.reverse()
    labels = [
        f"{', '.join(plan['hired']) if plan['hired'] else 'nobody'}   ·   cost {_tick(plan['cost'])}"
        for plan, _probability in ranked
    ]
    values = [probability for _plan, probability in ranked]
    axes[1].barh(np.arange(len(ranked)), values, color=[_plan_color(plan) for plan, _probability in ranked], height=0.72)
    axes[1].set_yticks(np.arange(len(ranked)), labels, fontsize=9)
    axes[1].set_xlabel("Probability")
    axes[1].set_xlim(0, max(values) * 1.22)
    axes[1].set_title("Eight most likely plans", loc="left")
    for index, probability in enumerate(values):
        axes[1].text(probability + 0.004, index, f"{probability:.2f}", va="center", fontsize=9, color=INK)
    figure.legend(
        handles=_plan_handles(),
        frameon=False,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.0),
    )
    figure.tight_layout(rect=(0, 0.08, 1, 1))
    return _save(figure, path)


def _solution(payload: dict, path: Path) -> Path:
    names = payload["names"]
    hired = set(payload["optimal_hired"])
    channels = payload["channels"]
    used = set()
    for person, source in channels.items():
        if source is not None and source != person:
            used.add(_edge_key(person, source))
    edge_color = {}
    edge_width = {}
    for person, friend in payload["edges"]:
        key = _edge_key(person, friend)
        if key in used:
            edge_color[key] = CRIMSON
            edge_width[key] = 2.8
        else:
            edge_color[key] = "#d5d8df"
            edge_width[key] = 1.4
    face = {name: (CRIMSON if name in hired else PALE) for name in names}
    border = {name: (CRIMSON if name in hired else BLUE) for name in names}

    figure, ax = plt.subplots(figsize=(7.6, 4.8))
    _draw_graph(
        ax,
        names,
        payload["edges"],
        _advertising_positions(),
        node_face=face,
        edge_color=edge_color,
        edge_width=edge_width,
        node_edge=border,
    )
    ax.set_title("Hiring plan of cost 2", loc="left")
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=CRIMSON, markeredgecolor=CRIMSON, ms=10, label="Hired"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=PALE, markeredgecolor=BLUE, ms=10, label="Reached through a friend"),
        Line2D([0], [0], color=CRIMSON, lw=2.4, label="Friendship that carries the product"),
        Line2D([0], [0], color="#d5d8df", lw=1.6, label="Friendship not used by this plan"),
    ]
    ax.legend(handles=handles, frameon=False, loc="upper right", fontsize=8.5)
    _pad_graph(ax)
    return _save(figure, path)


def _operating_depth(payload: dict) -> dict:
    for record in payload["depths"]:
        if record["depth"] == 3:
            return record
    return payload["depths"][-1]
