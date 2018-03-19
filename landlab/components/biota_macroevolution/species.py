"""Species BiotaEvolver object.
"""

from landlab.components.biota_macroevolution import (BiotaEvolverObject,
                                                     HabitatPatchVector)
import numpy as np
from random import random
from uuid import uuid4


class Species(BiotaEvolverObject):
    """A BiotaEvolver species.

    Species contains

    A universally unique identifier (UUID) is assigned to the species at
    initialization. The id is passed to child species.
    """

    def __init__(self, initial_time, initial_habitat_patches,
                 parent_species_id=-1):
        """Initialize a species.

        Parameters
        ----------
        initial_time : float
            Initial time of the species.
        initial_habitat_patches : HabitatPatch list
            A list of BiotaEvolver HabitatPatch objects of the species at the
            initial time.
        parent_species_id : UUID
            The identifier of the parent species. An id of -1 indicates no
            parent species.
        """
        BiotaEvolverObject.__init__(self)

        # Set parameters.
        self.parent_species_id = parent_species_id
        self.identifier = uuid4()

        # Set initial patch(es).
        if isinstance(initial_habitat_patches, list):
            p = initial_habitat_patches
        else:
            p = [initial_habitat_patches]
        self.record[initial_time] = {'habitat_patches': p}

    def run_macroevolution_processes(self, time, habitat_patch_vectors):
        """ Run disperal, speciation, and extinction processes.

        Parameters
        ----------
        time : float

        habitat_patch_vectors : BiotaEvolver HabitatPatch list

        Returns
        -------
        surviving_species : BiotaEvolver Species list
            The species that exist after the macroevolution processes run. This
            may include self and/or child species of self, or None if no
            species survive.
        """
        extant_species = []

        # Disperse and speciate.

        for v in habitat_patch_vectors:
            if v.cardinality in [HabitatPatchVector.ONE_TO_ONE,
                                 HabitatPatchVector.MANY_TO_ONE]:

                self.record[time] = {'habitat_patches': v.destinations}
                extant_species.append(self)

            elif v.cardinality in [HabitatPatchVector.ONE_TO_MANY,
                                   HabitatPatchVector.MANY_TO_MANY]:

                for d in v.destinations:
                    extant_species.append(Species(time, d, parent_species_id=
                                                 self.identifier))

        extant_species = np.array(list(set(extant_species)))

        # Evaluate extinction.

        extinction_chance = 0.10

        survival_probability = np.random.choice(np.linspace(0, 1, 100),
                                                len(extant_species))
        survival_results = survival_probability > extinction_chance
        surviving_species = list(extant_species[survival_results])

        return surviving_species

    def disperse(self, destination_patches, grid):
        self.habitat_patches = destination_patches

    def speciate(self):
        pass
