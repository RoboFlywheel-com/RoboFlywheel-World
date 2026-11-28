# SPDX-License-Identifier: Apache-2.0
"""The spec's palette decides the room's surfaces.

The defect: `StylePlan.palette` was parsed, validated and hashed, and read by nothing -- the room's
materials were four module constants in the render worker. Two specs asking for different colour
families produced byte-identical rooms. These pin the resolution rules, including the two the first
version got wrong on the shipped experiments: a token serves one role, and a token named for a role
beats one that merely tolerates it.
"""

from __future__ import annotations

import pytest

from rfw_w.scenes.contracts.scene_spec import StylePlan
from rfw_w.scenes.solvers.materials import (
    ACCENT,
    FLOOR,
    FURNITURE,
    PALETTE_LIBRARY,
    WALL,
    RoomMaterials,
    Surface,
    materials_for,
)

OFFICE = StylePlan(theme="warm_wood_office",
                   palette=("oak", "off_white", "muted_green", "brass"))
KITCHEN = StylePlan(theme="bright_kitchen", palette=("off_white", "oak", "sage", "steel"))
BEDROOM = StylePlan(theme="warm_wood_bedroom", palette=("oak", "linen", "sage", "brass"))


class TestOneTokenPerRole:
    def test_the_floor_claims_its_token_before_the_wall_does(self) -> None:
        """`oak` may serve both, and the first version let the wall take it.

        The office palette lists `oak` first, so the walls came out wood-brown and the `off_white`
        it asked for went unused. The floor is what the room is stood on and claims first.
        """
        materials = materials_for(OFFICE, seed=11)
        assert materials.floor.colour == PALETTE_LIBRARY["oak"].surface.colour
        assert materials.wall.colour == PALETTE_LIBRARY["off_white"].surface.colour

    def test_a_token_named_for_a_role_beats_one_that_tolerates_it(self) -> None:
        """`brass` exists only to be an accent; `muted_green` is a wall that can also be one.

        Without this the office's door frames came out muted green and the `brass` the palette
        asked for was read and then ignored.
        """
        materials = materials_for(OFFICE, seed=11)
        assert materials.frame.colour == PALETTE_LIBRARY["brass"].surface.colour
        assert materials.frame.metallic == PALETTE_LIBRARY["brass"].surface.metallic

    def test_the_ceiling_is_the_wall_lightened(self) -> None:
        """Not a fourth token: a ceiling is the wall colour, lighter."""
        materials = materials_for(OFFICE, seed=11)
        assert materials.ceiling.colour != materials.wall.colour
        for ceiling, wall in zip(materials.ceiling.colour, materials.wall.colour):
            assert ceiling >= wall

    def test_the_three_shipped_experiments_come_out_different(self) -> None:
        """The whole point: three palettes, three rooms."""
        rooms = [materials_for(style, seed=seed)
                 for style, seed in ((OFFICE, 11), (KITCHEN, 23), (BEDROOM, 37))]
        assert rooms[0].wall.colour != rooms[2].wall.colour  # off_white against linen
        assert rooms[0].frame.colour != rooms[1].frame.colour  # brass against steel
        assert len({room.to_dict()["wall"]["colour"][0] for room in rooms}) > 1


class TestWhatIsRecorded:
    def test_an_unknown_token_is_reported_not_defaulted(self) -> None:
        materials = materials_for(StylePlan(theme="x", palette=("chartreuse", "oak")), seed=1)
        assert materials.unmapped_tokens == ("chartreuse",)
        # And the tokens that *are* known still resolve.
        assert materials.floor.colour == PALETTE_LIBRARY["oak"].surface.colour

    def test_a_role_the_palette_names_nothing_for_is_recorded(self) -> None:
        """A palette of one metal names nothing for the walls, the floor or the furniture.

        `furniture` is in the set rather than absent: a brass-only palette has no wood to give the
        desk this pipeline builds, so the desk falls back and says so. The alternative -- borrowing
        brass for it -- would be recorded too, but as a reuse; what must never happen is a surface
        that fell back reading as one that was chosen.
        """
        materials = materials_for(StylePlan(theme="x", palette=("brass",)), seed=1)
        assert set(materials.roles_defaulted) == {WALL, FLOOR, FURNITURE}
        assert ACCENT not in materials.roles_defaulted
        assert materials.palette_reuses == ()

    def test_a_resolved_room_records_which_tokens_it_used(self) -> None:
        materials = materials_for(OFFICE, seed=11)
        assert set(materials.palette_used) == {"oak", "off_white", "brass"}
        assert materials.roles_defaulted == ()


class TestDeterminism:
    def test_the_same_request_gives_the_same_room(self) -> None:
        assert materials_for(OFFICE, seed=11) == materials_for(OFFICE, seed=11)

    def test_the_seed_moves_the_finish_and_not_the_colour(self) -> None:
        """The palette decides colour, the seed decides finish. Neither does the other's job."""
        first = materials_for(OFFICE, seed=1)
        second = materials_for(OFFICE, seed=2)
        assert first.wall.colour == second.wall.colour
        assert first.wall.roughness != second.wall.roughness

    def test_the_round_trip_is_exact(self) -> None:
        materials = materials_for(BEDROOM, seed=37)
        assert RoomMaterials.from_dict(materials.to_dict()) == materials


class TestTheDefaults:
    def test_they_are_the_colours_the_renderer_used_to_hardcode(self) -> None:
        """A request built without materials must render exactly as it did before this existed."""
        from rfw_w.scenes.workers.isaac.protocol import RoomMaterialsRequest

        default = RoomMaterials()
        request = RoomMaterialsRequest()
        assert tuple(default.wall.colour) == tuple(request.wall.colour) == (0.82, 0.79, 0.74)
        assert tuple(default.floor.colour) == tuple(request.floor.colour) == (0.42, 0.30, 0.21)
        assert tuple(default.ceiling.colour) == tuple(request.ceiling.colour) == (0.90, 0.89, 0.87)
        assert tuple(default.facade.colour) == tuple(request.facade.colour) == (0.55, 0.68, 0.85)

    def test_a_surface_refuses_a_colour_outside_the_unit_range(self) -> None:
        with pytest.raises(ValueError, match="within"):
            Surface((1.4, 0.5, 0.5), 0.5)

    def test_a_surface_refuses_a_roughness_outside_the_unit_range(self) -> None:
        with pytest.raises(ValueError, match="roughness"):
            Surface((0.5, 0.5, 0.5), 1.5)
