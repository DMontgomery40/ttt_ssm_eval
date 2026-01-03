/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        // Dark mode first color scheme
        surface: {
          DEFAULT: '#0d1117',
          50: '#161b22',
          100: '#21262d',
          200: '#30363d',
          300: '#484f58',
        },
        accent: {
          blue: '#58a6ff',
          green: '#238636',
          red: '#da3633',
          orange: '#d29922',
          purple: '#a371f7',
          cyan: '#39d353',
        },
        text: {
          primary: '#e6edf3',
          secondary: '#8b949e',
          muted: '#6e7681',
        }
      },
      fontFamily: {
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'glow': 'glow 2s ease-in-out infinite alternate',
      },
      keyframes: {
        glow: {
          '0%': { boxShadow: '0 0 5px rgba(88, 166, 255, 0.5)' },
          '100%': { boxShadow: '0 0 20px rgba(88, 166, 255, 0.8)' },
        }
      }
    },
  },
  plugins: [],
}
