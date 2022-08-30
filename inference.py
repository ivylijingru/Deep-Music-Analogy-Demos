import numpy as np
import torch
import pretty_midi as pm

from ec2vae.model import EC2VAE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def note_array_to_onehot(note_array):
    pr = np.zeros((len(note_array), 130))
    pr[np.arange(0, len(note_array)), note_array.astype(int)] = 1.
    return pr

def generate_midi_with_melody_chord(fn, mel_notes, c_notes):
    midi = pm.PrettyMIDI()
    ins1 = pm.Instrument(0)
    ins1.notes = mel_notes
    ins2 = pm.Instrument(0)
    ins2.notes = c_notes
    midi.instruments.append(ins1)
    midi.instruments.append(ins2)
    midi.write(fn)

'''init pitch & rhythm representation'''
def init_note():
    # x1: "From the new world" melody
    x1 = np.array([64, 128, 128, 67, 67, 128, 128, 128, 64, 128, 128, 62, 60, 128, 128, 128,
                62, 128, 128, 64, 67, 128, 128, 64, 62, 128, 128, 128, 129, 129, 129, 129])

    # x2: C4, sixteenth notes.
    x2 = np.array([60] * 32)

    pr1 = note_array_to_onehot(x1)
    pr2 = note_array_to_onehot(x2)

    pr1 = torch.from_numpy(pr1).float().to(device).unsqueeze(0)
    pr2 = torch.from_numpy(pr2).float().to(device).unsqueeze(0)

    print(pr1.size(), pr2.size())

    return pr1, pr2

def init_chord():
    # some useful chords.
    amin = [1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0]
    gmaj = [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1]
    fmaj = [1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0]
    emin = [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1]
    cmaj = [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]
    cmin = [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]

    # c1: Cmaj - - - | Gmaj - - - ||
    c1 = np.array([cmaj] * 16 + [gmaj] * 16)
    # c2: Amin - Gmaj - | Fmaj - Emin - ||
    c2 = np.array([amin] * 8 + [gmaj] * 8 + [fmaj] * 8 + [emin] * 8)
    # no chord
    c3 = np.zeros((32, 12))

    c1 = torch.from_numpy(c1).float().to(device).unsqueeze(0)
    c2 = torch.from_numpy(c2).float().to(device).unsqueeze(0)
    c3 = torch.from_numpy(c3).float().to(device).unsqueeze(0)

    return c1, c2, c3

def swap_rhythm_chord(ec2vae_model):
    pr1, pr2 = init_note()
    c1, c2, c3 = init_chord()

    dist_p1, dist_r1 = ec2vae_model.encoder(pr1, c1)
    zp1 = dist_p1.sample()
    zr1 = dist_r1.sample()
    
    dist_p2, dist_r2 = ec2vae_model.encoder(pr2, c3)
    zp2 = dist_p2.sample()
    zr2 = dist_r2.sample()

    print(zp1.size(), zr1.size(), zp2.size(), zr2.size())

    with torch.no_grad():
        pred_recon = ec2vae_model.decoder(zp1, zr1, c1)
        pred_new_rhythm = ec2vae_model.decoder(zp1, zr2, c1)
        pred_new_chord = ec2vae_model.decoder(zp1, zr1, c2)

    print(pred_recon.size(), pred_new_rhythm.size(), pred_new_chord.size())

    out_recon = pred_recon.squeeze(0).cpu().numpy()
    out_new_rhythm = pred_new_rhythm.squeeze(0).cpu().numpy()
    out_new_chord = pred_new_chord.squeeze(0).cpu().numpy()
    
    print(out_recon.shape, out_new_rhythm.shape, out_new_chord.shape)

    notes_recon = ec2vae_model.__class__.note_array_to_notes(out_recon, bpm=120, start=0.)
    notes_new_rhythm = ec2vae_model.__class__.note_array_to_notes(out_new_rhythm, bpm=120, start=0.)
    notes_new_chord = ec2vae_model.__class__.note_array_to_notes(out_new_chord, bpm=120, start=0.)

    notes_c1 = ec2vae_model.__class__.chord_to_notes(c1.squeeze(0).cpu().numpy(), 120, 0)
    notes_c2 = ec2vae_model.__class__.chord_to_notes(c2.squeeze(0).cpu().numpy(), 120, 0)

    generate_midi_with_melody_chord('./demo/ec2vae-recon.mid', notes_recon, notes_c1)
    generate_midi_with_melody_chord('./demo/ec2vae-new-rhythm.mid', notes_new_rhythm, notes_c1)
    generate_midi_with_melody_chord('./demo/ec2vae-new-chord.mid', notes_new_chord, notes_c2)

if __name__ == "__main__":
    # initialize the model
    ec2vae_model = EC2VAE.init_model()

    # load model parameter
    ec2vae_param_path = 'model_parameters.pt'
    ec2vae_model.load_model(ec2vae_param_path)
    ec2vae_model.eval()

    swap_rhythm_chord(ec2vae_model)
