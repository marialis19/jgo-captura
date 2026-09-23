import fs from "node:fs";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  server: {
    host: "0.0.0.0",
    https: {
      key: fs.readFileSync("./localhost+2-key.pem"),
      cert: fs.readFileSync("./localhost+2.pem")
    }
  }
});