import drone_config
from tello_otonom import OtonomSistem

drone_config.SAGA_SOLA_MESAFE      = 50
drone_config.ILERI_GITME_MESAFE    = 50
drone_config.GERI_GITME_MESAFE     = 80
drone_config.HEDEF_IDEAL_MESAFE_CM = 40
drone_config.YUKARI_GITME_MESAFE   = 80
drone_config.ASAGI_GITME_MESAFE    = 50
drone_config.DONUS_ACISI           = 90

drone_config.AI_GUVEN_ESIGI = 0.95
drone_config.FIRE_CONF      = 0.60
drone_config.SMOKE_CONF     = 0.45
drone_config.AI_IMG_SIZE    = 640

drone = OtonomSistem()

@drone.hedefte("sol")
def sol_git(tello):
    print("Sola gidiyorum")
    tello.move_left(drone_config.SAGA_SOLA_MESAFE)

if __name__ == "__main__":
    drone.baslat()
