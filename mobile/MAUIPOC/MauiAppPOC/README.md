# .NET MAUI POC

This is a proof of concept (POC) .NET MAUI application demonstrating cross-platform mobile and desktop development.

## Features

This POC demonstrates:

### 1. **Counter Demo**
   - Simple button click counter with state management
   - Accessibility support with semantic screen reader

### 2. **Data Binding**
   - Real-time text input binding
   - Dynamic label updates based on user input

### 3. **Navigation**
   - Shell-based navigation between pages
   - Passing parameters between pages using query strings
   - Back navigation

### 4. **Platform Information**
   - Displays current platform, device model, and OS version
   - Cross-platform device detection

### 5. **Device Features**
   - Battery status monitoring
   - Network connectivity checking
   - Access to native device APIs

### 6. **Styling & Theming**
   - Light/Dark mode support
   - Custom color schemes
   - Reusable styles across the app

## Project Structure

```
MauiAppPOC/
├── App.xaml & App.xaml.cs          # Application entry point
├── AppShell.xaml & AppShell.xaml.cs # Shell navigation configuration
├── MauiProgram.cs                   # MAUI app builder and configuration
├── MainPage.xaml & MainPage.xaml.cs # Main page with POC features
├── DetailsPage.xaml & DetailsPage.xaml.cs # Details page for navigation demo
├── Platforms/                       # Platform-specific code
│   ├── Android/                     # Android-specific implementations
│   ├── iOS/                         # iOS-specific implementations
│   └── MacCatalyst/                 # macOS-specific implementations
└── Resources/                       # App resources
    ├── AppIcon/                     # App icon assets
    ├── Fonts/                       # Custom fonts
    ├── Images/                      # Image assets
    ├── Splash/                      # Splash screen
    └── Styles/                      # XAML styles
        ├── Colors.xaml              # Color definitions
        └── Styles.xaml              # UI control styles
```

## Prerequisites

To build and run this project, you need:

1. **.NET 9.0 SDK** or later
2. **.NET MAUI workload** installed:
   ```bash
   sudo dotnet workload install maui
   ```

### Platform-Specific Requirements

- **Android**: Android SDK with API level 21+
- **iOS/macOS**: Xcode 13+ (macOS only)
- **Windows**: Visual Studio 2022 with Windows App SDK

## Building the Project

```bash
# Restore dependencies
dotnet restore

# Build for specific platform
dotnet build -f net9.0-android
dotnet build -f net9.0-ios
dotnet build -f net9.0-maccatalyst

# Run on specific platform
dotnet run -f net9.0-android
dotnet run -f net9.0-ios
```

## Key Technologies

- **.NET MAUI**: Cross-platform UI framework
- **XAML**: UI markup language
- **C#**: Application logic
- **Shell Navigation**: Modern navigation pattern
- **Platform APIs**: Battery, Connectivity, DeviceInfo

## Next Steps

To extend this POC, consider adding:

- [ ] MVVM pattern with data binding
- [ ] Dependency injection
- [ ] Local database (SQLite)
- [ ] REST API integration
- [ ] Authentication
- [ ] Push notifications
- [ ] More complex UI layouts
- [ ] Unit tests

## Resources

- [.NET MAUI Documentation](https://docs.microsoft.com/dotnet/maui)
- [MAUI Samples](https://github.com/dotnet/maui-samples)
- [MAUI Community Toolkit](https://github.com/CommunityToolkit/Maui)
