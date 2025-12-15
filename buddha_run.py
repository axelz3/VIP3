import numpy as np  
import numpy.linalg as la  
import matplotlib.pyplot as plt  
from pathlib import Path  
import ps_utils 

OUTPUT_DIR = Path(__file__).resolve().parent  # save everything next to this script


def build_J(I, mask):
    """
    I: (m, n, k)
    mask: (m, n) boolean/0-1
    Returns:
      J: (k, nz) where nz = number of pixels in mask
      inside: tuple of indices (rows, cols) for masked pixels
    """
    inside = np.where(mask > 0)  # indices of valid pixels
    J = I[inside[0], inside[1], :].T  # gather light stack only on valid pixels
    return J, inside


def unpack_M_to_fields(M, mask, inside, eps=1e-8):
    """
    M: (3, nz) = rho*n
    Returns: albedo_img, (n1, n2, n3) as (m, n) arrays (outside mask filled with 0/1)
    """
    m, n = mask.shape
    rho = la.norm(M, axis=0)  # albedo is vector length
    N = M / (rho + eps)  # normals are unit vectors

    albedo_img = np.zeros((m, n), dtype=float)
    albedo_img[inside] = rho  # write values only where mask was true

    n1 = np.zeros((m, n), dtype=float)
    n2 = np.zeros((m, n), dtype=float)
    n3 = np.ones((m, n), dtype=float)  # default normal z component

    n1[inside] = N[0, :]  # drop components back into image form
    n2[inside] = N[1, :]
    n3[inside] = N[2, :]

    return albedo_img, (n1, n2, n3)


def save_ps_result(name_prefix, albedo, normals, depth):
    """Persist the main outputs for later inspection."""
    n1, n2, n3 = normals  # unpack tuple
    out_path = OUTPUT_DIR / f"{name_prefix}_outputs.npz"  # build save name
    np.savez(out_path, albedo=albedo, n1=n1, n2=n2, n3=n3, depth=depth)  # dump arrays
    print(f"Saved results to {out_path}")  # quick confirmation


def show_albedo_and_normals(albedo, n1, n2, n3, title_prefix="", save_path=None):
    fig, ax = plt.subplots(1, 4, figsize=(14, 4))  # layout for albedo + normals
    ax[0].imshow(albedo); ax[0].set_title(f"{title_prefix}albedo")
    ax[1].imshow(n1);     ax[1].set_title(f"{title_prefix}n1")
    ax[2].imshow(n2);     ax[2].set_title(f"{title_prefix}n2")
    ax[3].imshow(n3);     ax[3].set_title(f"{title_prefix}n3")
    for a in ax: a.axis("off")
    plt.tight_layout()  # keep spacing clean
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")  # write figure if path given
        print(f"Saved figure to {save_path}")
    plt.show()  # still show on screen
    plt.close(fig)  # free memory


def estimate_M_pinv(S, J):
    # M = S^† J  (S is (k,3), J is (k,nz)) -> M is (3,nz)
    return la.pinv(S) @ J


def estimate_M_ransac(S, J, threshold, verbose_every=2000):
    """
    RANSAC must be done pixel-by-pixel. :contentReference[oaicite:1]{index=1}
    Returns M: (3, nz)
    """
    k, nz = J.shape  # number of lights and pixels
    M = np.zeros((3, nz), dtype=float)  # placeholder for rho*n at every pixel

    for p in range(nz):
        Ivec = J[:, p]  # intensity vector for one pixel
        out = ps_utils.ransac_3dvector((Ivec, S), threshold=threshold, verbose=0)  # robust fit
        if out is not None:
            m_vec, inliers, best_fit = out
            M[:, p] = m_vec  # store the estimated rho*n

        if verbose_every and (p % verbose_every == 0):
            print(f"RANSAC: {p}/{nz}")

    return M


def main():
    # Load dataset
    I, mask, S = ps_utils.read_data_file("Buddha")  # same style as the provided Beethoven runner 

    J, inside = build_J(I, mask)  # reshape data to matrix form

    # --- (A) Woodham / pseudo-inverse ---
    M_pinv = estimate_M_pinv(S, J)  # basic least-squares solution
    albedo_p, (n1_p, n2_p, n3_p) = unpack_M_to_fields(M_pinv, mask, inside)  # convert back to images
    show_albedo_and_normals(
        albedo_p,
        n1_p,
        n2_p,
        n3_p,
        title_prefix="Pseudo-inverse: ",
        save_path=OUTPUT_DIR / "buddha_Pseudo_inverse_normals.png",
    )

    z_p = ps_utils.unbiased_integrate(n1_p, n2_p, n3_p, mask)  # integrate normals into depth
    z_p = np.nan_to_num(z_p)  # replace invalid values
    save_ps_result("buddha_pseudo_inverse", albedo_p, (n1_p, n2_p, n3_p), z_p)  # save result pack
    ps_utils.display_surface(z_p, albedo=albedo_p)  # render mesh

    # --- (B) RANSAC (threshold >= 25) ---
    thr = 50.0  # outlier threshold for RANSAC, can be tuned
    M_r = estimate_M_ransac(S, J, threshold=thr)  # robust estimation for each pixel
    albedo_r, (n1_r, n2_r, n3_r) = unpack_M_to_fields(M_r, mask, inside)  # map back to images
    show_albedo_and_normals(
        albedo_r,
        n1_r,
        n2_r,
        n3_r,
        title_prefix=f"ransac(thr={thr}): ",
        save_path=OUTPUT_DIR / "buddha_ransac_normals.png",
    )

    z_r = ps_utils.unbiased_integrate(n1_r, n2_r, n3_r, mask)  # integrate normals
    z_r = np.nan_to_num(z_r)  # clean depth map
    save_ps_result("buddha_ransac", albedo_r, (n1_r, n2_r, n3_r), z_r)  # save outputs
    ps_utils.display_surface(z_r, albedo=albedo_r)  # render new mesh

    # --- (C) Smooth the RANSAC normal field ---
    n1_s, n2_s, n3_s = ps_utils.smooth_normal_field(n1_r, n2_r, n3_r, mask, iters=200, tau=0.05, verbose=True)  # diffuse normals
    show_albedo_and_normals(
        albedo_r,
        n1_s,
        n2_s,
        n3_s,
        title_prefix="ransac + smooth: ",
        save_path=OUTPUT_DIR / "buddha_ransac_smooth_normals.png",
    )

    z_s = ps_utils.unbiased_integrate(n1_s, n2_s, n3_s, mask)  # integrate smoothed normals
    z_s = np.nan_to_num(z_s)  # clean depth map
    save_ps_result("buddha_ransac_smooth", albedo_r, (n1_s, n2_s, n3_s), z_s)  # save pack
    ps_utils.display_surface(z_s, albedo=albedo_r)  # final render


if __name__ == "__main__":
    main()
