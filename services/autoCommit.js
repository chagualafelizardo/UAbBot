import { exec } from "child_process";

const commitMessage = `Auto-commit: ${new Date().toISOString()}`;

exec("git add .", (error, stdout, stderr) => {
  if (error) {
    console.error(`Error adding files: ${error.message}`);
    return;
  }

  exec(`git commit -m "${commitMessage}"`, (error, stdout, stderr) => {
    if (error) {
      console.error(`Error committing files: ${error.message}`);
      return;
    }

    exec("git push origin main", (error, stdout, stderr) => {
      if (error) {
        console.error(`Error pushing to GitHub: ${error.message}`);
        return;
      }
      console.log("✅ Successfully committed and pushed to GitHub!");
    });
  });
});
